from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
import re


class RankSourcesInput(BaseModel):
    sources: List[Dict[str, Any]] = Field(description="List of sources with url, title, content")
    query: str = Field(description="Original search query for relevance scoring")
    prioritize_recent: bool = Field(default=True, description="Prioritize recent sources")


class CacheQueryInput(BaseModel):
    query: str = Field(description="Search query to cache")
    results: List[Dict[str, Any]] = Field(description="Search results to cache")
    ttl_hours: int = Field(default=24, description="Time to live in hours")


class GetCachedInput(BaseModel):
    query: str = Field(description="Search query to retrieve from cache")


class UpdateTrustScoreInput(BaseModel):
    domain: str = Field(description="Domain to update trust score for")
    new_score: float = Field(description="New trust score (0-100)")
    reason: str = Field(default="", description="Reason for score update")


# Global storage
_source_rankings: Dict[str, Dict[str, Any]] = {}
_query_cache: Dict[str, Dict[str, Any]] = {}
_trust_scores: Dict[str, float] = {}
_cache_file = Path("source_cache.json")

# Base trust scores for different source types
BASE_TRUST_SCORES = {
    # Academic and Research (90-100)
    "scholar.google.com": 95,
    "pubmed.ncbi.nlm.nih.gov": 95,
    "arxiv.org": 90,
    "jstor.org": 90,
    "nature.com": 95,
    "science.org": 95,
    "cell.com": 90,
    "plos.org": 85,
    "springer.com": 85,
    "wiley.com": 85,
    "elsevier.com": 85,
    "ieee.org": 90,
    "acm.org": 90,
    
    # Government and Official (85-95)
    "gov": 90,
    "edu": 85,
    "who.int": 90,
    "cdc.gov": 90,
    "nih.gov": 90,
    "fda.gov": 85,
    "nasa.gov": 85,
    "noaa.gov": 85,
    "usgs.gov": 85,
    "europa.eu": 85,
    "un.org": 85,
    
    # Reputable News (70-85)
    "reuters.com": 85,
    "bbc.com": 85,
    "apnews.com": 85,
    "npr.org": 80,
    "theguardian.com": 75,
    "nytimes.com": 75,
    "washingtonpost.com": 75,
    "wsj.com": 75,
    "economist.com": 80,
    "ft.com": 75,
    "bloomberg.com": 75,
    
    # Fact-checking sites (75-85)
    "snopes.com": 80,
    "factcheck.org": 85,
    "politifact.com": 80,
    "fullfact.org": 80,
    "factchecker.in": 75,
    
    # Reference and Encyclopedia (65-80)
    "wikipedia.org": 70,
    "britannica.com": 80,
    "merriam-webster.com": 75,
    "dictionary.com": 70,
    
    # Professional Organizations (70-85)
    "ieee.org": 85,
    "acm.org": 85,
    "ama-assn.org": 80,
    "apa.org": 80,
    
    # Tech and Industry (60-75)
    "techcrunch.com": 65,
    "wired.com": 70,
    "arstechnica.com": 70,
    "stackoverflow.com": 75,
    "github.com": 70,
    
    # General domains (lower scores)
    "medium.com": 50,
    "wordpress.com": 40,
    "blogspot.com": 35,
    "tumblr.com": 30,
    
    # Social media (very low scores)
    "twitter.com": 25,
    "facebook.com": 25,
    "instagram.com": 20,
    "tiktok.com": 15,
    "reddit.com": 40,  # Slightly higher due to community moderation
}

# Source type classifications
SOURCE_TYPE_SCORES = {
    "academic": 90,
    "government": 85,
    "news": 70,
    "organization": 65,
    "reference": 70,
    "blog": 40,
    "social": 25,
    "unknown": 30
}

# Quality indicators that boost trust scores
QUALITY_INDICATORS = {
    "peer_reviewed": 15,
    "has_doi": 10,
    "has_citations": 8,
    "author_credentials": 5,
    "recent_publication": 5,
    "methodology_described": 8,
    "data_available": 5,
    "institutional_affiliation": 5
}

# Red flags that reduce trust scores
RED_FLAGS = {
    "clickbait_title": -15,
    "no_author": -10,
    "no_date": -8,
    "broken_links": -5,
    "poor_grammar": -10,
    "conspiracy_language": -20,
    "unsubstantiated_claims": -15,
    "promotional_content": -10
}


def _load_cache():
    """Load cache from file."""
    global _query_cache, _trust_scores
    
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r') as f:
                data = json.load(f)
                _query_cache = data.get("query_cache", {})
                _trust_scores = data.get("trust_scores", {})
        except Exception:
            _query_cache = {}
            _trust_scores = {}


def _save_cache():
    """Save cache to file."""
    try:
        data = {
            "query_cache": _query_cache,
            "trust_scores": _trust_scores,
            "last_updated": datetime.now().isoformat()
        }
        with open(_cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _get_domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return url.lower()


def _classify_source_type(url: str, title: str = "", content: str = "") -> str:
    """Classify the type of source."""
    url_lower = url.lower()
    title_lower = title.lower()
    content_lower = content.lower()
    
    # Academic sources
    if any(indicator in url_lower for indicator in [
        "scholar.google", "pubmed", "arxiv", "jstor", "nature.com", 
        "science.org", "ieee.org", "acm.org"
    ]):
        return "academic"
    
    if ".edu" in url_lower or any(word in title_lower for word in [
        "journal", "research", "study", "university"
    ]):
        return "academic"
    
    # Government sources
    if ".gov" in url_lower or any(domain in url_lower for domain in [
        "who.int", "europa.eu", "un.org"
    ]):
        return "government"
    
    # News sources
    if any(domain in url_lower for domain in [
        "reuters", "bbc", "cnn", "npr", "guardian", "nytimes", 
        "washingtonpost", "apnews", "bloomberg", "economist"
    ]):
        return "news"
    
    # Social media
    if any(domain in url_lower for domain in [
        "twitter", "facebook", "instagram", "tiktok", "reddit"
    ]):
        return "social"
    
    # Blogs
    if any(indicator in url_lower for indicator in [
        "blog", "wordpress", "medium.com", "substack"
    ]):
        return "blog"
    
    # Organizations
    if ".org" in url_lower:
        return "organization"
    
    # Reference
    if any(domain in url_lower for domain in [
        "wikipedia", "britannica", "dictionary"
    ]):
        return "reference"
    
    return "unknown"


def _calculate_base_trust_score(url: str) -> float:
    """Calculate base trust score for a URL."""
    domain = _get_domain_from_url(url)
    
    # Check for exact domain matches
    for trusted_domain, score in BASE_TRUST_SCORES.items():
        if trusted_domain in domain:
            return score
    
    # Check for TLD-based scoring
    if domain.endswith('.edu'):
        return 85
    elif domain.endswith('.gov'):
        return 90
    elif domain.endswith('.org'):
        return 65
    elif domain.endswith('.com'):
        return 50
    else:
        return 40


def _analyze_content_quality(content: str, title: str = "") -> Dict[str, Any]:
    """Analyze content quality and return quality metrics."""
    content_lower = content.lower()
    title_lower = title.lower()
    
    quality_score = 0
    indicators = []
    red_flags = []
    
    # Check for quality indicators
    if "peer reviewed" in content_lower or "peer-reviewed" in content_lower:
        quality_score += QUALITY_INDICATORS["peer_reviewed"]
        indicators.append("peer_reviewed")
    
    if "doi:" in content_lower:
        quality_score += QUALITY_INDICATORS["has_doi"]
        indicators.append("has_doi")
    
    if re.search(r'\[\d+\]|\(\d{4}\)', content):
        quality_score += QUALITY_INDICATORS["has_citations"]
        indicators.append("has_citations")
    
    if "author" in content_lower and ("dr." in content_lower or "professor" in content_lower):
        quality_score += QUALITY_INDICATORS["author_credentials"]
        indicators.append("author_credentials")
    
    if "methodology" in content_lower or "methods" in content_lower:
        quality_score += QUALITY_INDICATORS["methodology_described"]
        indicators.append("methodology_described")
    
    # Check for red flags
    clickbait_patterns = [
        "you won't believe", "shocking", "amazing", "incredible",
        "doctors hate this", "secret", "trick"
    ]
    
    if any(pattern in title_lower for pattern in clickbait_patterns):
        quality_score += RED_FLAGS["clickbait_title"]
        red_flags.append("clickbait_title")
    
    if "conspiracy" in content_lower or "cover-up" in content_lower:
        quality_score += RED_FLAGS["conspiracy_language"]
        red_flags.append("conspiracy_language")
    
    # Content length and structure
    content_length = len(content)
    paragraph_count = len(content.split('\n\n'))
    
    return {
        "quality_score": quality_score,
        "indicators": indicators,
        "red_flags": red_flags,
        "content_length": content_length,
        "paragraph_count": paragraph_count,
        "has_structure": paragraph_count > 3
    }


def _calculate_relevance_score(query: str, title: str, content: str) -> float:
    """Calculate relevance score based on query match."""
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    content_words = set(content.lower().split())
    
    # Title relevance (weighted more heavily)
    title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
    
    # Content relevance
    content_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
    
    # Combined relevance score
    relevance_score = (title_overlap * 0.7) + (content_overlap * 0.3)
    
    return min(100, relevance_score * 100)


def _calculate_recency_score(content: str, url: str) -> float:
    """Calculate recency score based on publication date."""
    # Look for date patterns in content
    date_patterns = [
        r'\b(20\d{2})\b',  # Year
        r'\b(\d{1,2}/\d{1,2}/20\d{2})\b',  # MM/DD/YYYY
        r'\b(20\d{2}-\d{2}-\d{2})\b',  # YYYY-MM-DD
    ]
    
    current_year = datetime.now().year
    found_year = None
    
    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        if matches:
            try:
                if pattern == date_patterns[0]:  # Year only
                    found_year = int(matches[-1])  # Use most recent year found
                else:
                    # Extract year from date
                    date_str = matches[-1]
                    if '/' in date_str:
                        found_year = int(date_str.split('/')[-1])
                    elif '-' in date_str:
                        found_year = int(date_str.split('-')[0])
                break
            except (ValueError, IndexError):
                continue
    
    if found_year:
        years_old = current_year - found_year
        if years_old <= 1:
            return 100
        elif years_old <= 3:
            return 80
        elif years_old <= 5:
            return 60
        elif years_old <= 10:
            return 40
        else:
            return 20
    
    return 50  # Default if no date found


# Load cache on module import
_load_cache()


@tool(args_schema=RankSourcesInput)
def rank_sources_by_trust(
    sources: List[Dict[str, Any]], 
    query: str, 
    prioritize_recent: bool = True
) -> str:
    """
    Rank sources by trust score, relevance, and recency.
    
    Args:
        sources: List of sources with url, title, content
        query: Original search query for relevance scoring
        prioritize_recent: Whether to prioritize recent sources
    
    Returns:
        JSON string with ranked sources
    """
    try:
        ranked_sources = []
        
        for source in sources:
            url = source.get("url", "")
            title = source.get("title", "")
            content = source.get("content", "")
            
            # Calculate base trust score
            base_trust = _calculate_base_trust_score(url)
            
            # Get stored trust score if available
            domain = _get_domain_from_url(url)
            stored_trust = _trust_scores.get(domain, base_trust)
            
            # Analyze content quality
            quality_analysis = _analyze_content_quality(content, title)
            
            # Calculate relevance score
            relevance_score = _calculate_relevance_score(query, title, content)
            
            # Calculate recency score
            recency_score = _calculate_recency_score(content, url) if prioritize_recent else 50
            
            # Classify source type
            source_type = _classify_source_type(url, title, content)
            
            # Calculate final trust score
            final_trust_score = stored_trust + quality_analysis["quality_score"]
            final_trust_score = max(0, min(100, final_trust_score))
            
            # Calculate overall ranking score
            # Trust score is weighted most heavily, then relevance, then recency
            overall_score = (
                final_trust_score * 0.5 +
                relevance_score * 0.3 +
                recency_score * 0.2
            )
            
            ranked_source = {
                "url": url,
                "title": title,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "source_type": source_type,
                "trust_score": final_trust_score,
                "relevance_score": relevance_score,
                "recency_score": recency_score,
                "overall_score": overall_score,
                "quality_indicators": quality_analysis["indicators"],
                "red_flags": quality_analysis["red_flags"],
                "domain": domain
            }
            
            ranked_sources.append(ranked_source)
        
        # Sort by overall score (descending)
        ranked_sources.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Add ranking positions
        for i, source in enumerate(ranked_sources):
            source["rank"] = i + 1
        
        # Calculate ranking statistics
        avg_trust = sum(s["trust_score"] for s in ranked_sources) / len(ranked_sources) if ranked_sources else 0
        high_trust_count = len([s for s in ranked_sources if s["trust_score"] >= 80])
        
        result = {
            "status": "success",
            "ranked_sources": ranked_sources,
            "ranking_metadata": {
                "total_sources": len(ranked_sources),
                "average_trust_score": avg_trust,
                "high_trust_sources": high_trust_count,
                "query": query,
                "prioritize_recent": prioritize_recent,
                "ranking_timestamp": datetime.now().isoformat()
            },
            "message": f"Ranked {len(ranked_sources)} sources by trust score"
        }
        
        # Store ranking for future reference
        ranking_key = hashlib.md5(f"{query}{len(sources)}".encode()).hexdigest()
        _source_rankings[ranking_key] = result
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Source ranking failed: {str(e)}"
        })


@tool(args_schema=CacheQueryInput)
def cache_search_results(query: str, results: List[Dict[str, Any]], ttl_hours: int = 24) -> str:
    """
    Cache search results for frequent queries.
    
    Args:
        query: Search query to cache
        results: Search results to cache
        ttl_hours: Time to live in hours
    
    Returns:
        JSON string with caching result
    """
    try:
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        
        cache_entry = {
            "query": query,
            "results": results,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
            "ttl_hours": ttl_hours,
            "result_count": len(results)
        }
        
        _query_cache[cache_key] = cache_entry
        _save_cache()
        
        return json.dumps({
            "status": "success",
            "cache_key": cache_key,
            "cached_results": len(results),
            "expires_at": cache_entry["expires_at"],
            "message": f"Cached {len(results)} results for query: {query}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Caching failed: {str(e)}"
        })


@tool(args_schema=GetCachedInput)
def get_cached_results(query: str) -> str:
    """
    Retrieve cached search results for a query.
    
    Args:
        query: Search query to retrieve from cache
    
    Returns:
        JSON string with cached results or cache miss
    """
    try:
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        
        if cache_key in _query_cache:
            cache_entry = _query_cache[cache_key]
            expires_at = datetime.fromisoformat(cache_entry["expires_at"])
            
            if datetime.now() < expires_at:
                # Cache hit - return cached results
                return json.dumps({
                    "status": "cache_hit",
                    "query": query,
                    "results": cache_entry["results"],
                    "cached_at": cache_entry["cached_at"],
                    "expires_at": cache_entry["expires_at"],
                    "result_count": cache_entry["result_count"],
                    "message": f"Retrieved {cache_entry['result_count']} cached results"
                })
            else:
                # Cache expired - remove entry
                del _query_cache[cache_key]
                _save_cache()
        
        # Cache miss
        return json.dumps({
            "status": "cache_miss",
            "query": query,
            "message": "No cached results found for query"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Cache retrieval failed: {str(e)}"
        })


@tool(args_schema=UpdateTrustScoreInput)
def update_domain_trust_score(domain: str, new_score: float, reason: str = "") -> str:
    """
    Update trust score for a specific domain.
    
    Args:
        domain: Domain to update trust score for
        new_score: New trust score (0-100)
        reason: Reason for score update
    
    Returns:
        JSON string with update result
    """
    try:
        # Validate score range
        if not 0 <= new_score <= 100:
            return json.dumps({
                "status": "failure",
                "message": "Trust score must be between 0 and 100"
            })
        
        # Clean domain
        clean_domain = _get_domain_from_url(domain) if "://" in domain else domain.lower()
        
        old_score = _trust_scores.get(clean_domain, _calculate_base_trust_score(f"https://{clean_domain}"))
        
        _trust_scores[clean_domain] = new_score
        _save_cache()
        
        return json.dumps({
            "status": "success",
            "domain": clean_domain,
            "old_score": old_score,
            "new_score": new_score,
            "reason": reason,
            "updated_at": datetime.now().isoformat(),
            "message": f"Updated trust score for {clean_domain}: {old_score} → {new_score}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Trust score update failed: {str(e)}"
        })


@tool
def get_trust_score_rankings() -> str:
    """
    Get rankings of all domains by trust score.
    
    Returns:
        JSON string with domain trust rankings
    """
    try:
        # Combine base scores with custom scores
        all_scores = dict(BASE_TRUST_SCORES)
        all_scores.update(_trust_scores)
        
        # Sort by score (descending)
        sorted_domains = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize by score ranges
        categories = {
            "very_high_trust": [],  # 90-100
            "high_trust": [],       # 80-89
            "medium_trust": [],     # 60-79
            "low_trust": [],        # 40-59
            "very_low_trust": []    # 0-39
        }
        
        for domain, score in sorted_domains:
            if score >= 90:
                categories["very_high_trust"].append({"domain": domain, "score": score})
            elif score >= 80:
                categories["high_trust"].append({"domain": domain, "score": score})
            elif score >= 60:
                categories["medium_trust"].append({"domain": domain, "score": score})
            elif score >= 40:
                categories["low_trust"].append({"domain": domain, "score": score})
            else:
                categories["very_low_trust"].append({"domain": domain, "score": score})
        
        return json.dumps({
            "status": "success",
            "trust_rankings": {
                "total_domains": len(sorted_domains),
                "categories": categories,
                "top_10_domains": sorted_domains[:10],
                "average_score": sum(score for _, score in sorted_domains) / len(sorted_domains) if sorted_domains else 0
            },
            "message": f"Retrieved trust rankings for {len(sorted_domains)} domains"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get trust rankings: {str(e)}"
        })


@tool
def get_cache_statistics() -> str:
    """
    Get statistics about the query cache.
    
    Returns:
        JSON string with cache statistics
    """
    try:
        total_cached = len(_query_cache)
        
        if total_cached == 0:
            return json.dumps({
                "status": "success",
                "cache_statistics": {
                    "total_cached_queries": 0,
                    "message": "No queries cached yet"
                }
            })
        
        # Analyze cache entries
        now = datetime.now()
        active_entries = 0
        expired_entries = 0
        total_results = 0
        
        for entry in _query_cache.values():
            expires_at = datetime.fromisoformat(entry["expires_at"])
            if now < expires_at:
                active_entries += 1
                total_results += entry["result_count"]
            else:
                expired_entries += 1
        
        # Calculate cache efficiency
        cache_efficiency = (active_entries / total_cached * 100) if total_cached > 0 else 0
        
        return json.dumps({
            "status": "success",
            "cache_statistics": {
                "total_cached_queries": total_cached,
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "total_cached_results": total_results,
                "cache_efficiency": cache_efficiency,
                "average_results_per_query": total_results / active_entries if active_entries > 0 else 0
            },
            "message": f"Cache contains {total_cached} queries with {active_entries} active entries"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get cache statistics: {str(e)}"
        })


@tool
def clear_expired_cache() -> str:
    """
    Clear expired cache entries.
    
    Returns:
        JSON string with cleanup results
    """
    try:
        now = datetime.now()
        expired_keys = []
        
        for key, entry in _query_cache.items():
            expires_at = datetime.fromisoformat(entry["expires_at"])
            if now >= expires_at:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del _query_cache[key]
        
        if expired_keys:
            _save_cache()
        
        return json.dumps({
            "status": "success",
            "cleared_entries": len(expired_keys),
            "remaining_entries": len(_query_cache),
            "message": f"Cleared {len(expired_keys)} expired cache entries"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Cache cleanup failed: {str(e)}"
        })