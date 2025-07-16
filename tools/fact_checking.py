from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path


class FactCheckInput(BaseModel):
    claim: str = Field(description="Claim to fact-check")
    sources: List[str] = Field(description="List of source URLs or content to verify against")
    context: str = Field(default="", description="Additional context for the claim")


class CrossVerifyInput(BaseModel):
    claims: List[str] = Field(description="List of claims to cross-verify")
    source_urls: List[str] = Field(description="List of source URLs to check against")
    verification_threshold: float = Field(default=0.7, description="Threshold for verification confidence")


class SourceCredibilityInput(BaseModel):
    url: str = Field(description="URL to evaluate for credibility")
    content: str = Field(default="", description="Content from the source")
    title: str = Field(default="", description="Title of the source")


# Global fact-checking storage
_fact_check_cache: Dict[str, Dict[str, Any]] = {}
_credibility_scores: Dict[str, float] = {}

# Trusted source patterns
TRUSTED_DOMAINS = {
    # Academic and Research
    "scholar.google.com": 95,
    "pubmed.ncbi.nlm.nih.gov": 95,
    "arxiv.org": 90,
    "jstor.org": 90,
    "nature.com": 95,
    "science.org": 95,
    "cell.com": 90,
    "plos.org": 85,
    
    # Government and Official
    "gov": 90,
    "edu": 85,
    "who.int": 90,
    "cdc.gov": 90,
    "nih.gov": 90,
    "fda.gov": 85,
    "nasa.gov": 85,
    
    # Reputable News
    "reuters.com": 85,
    "bbc.com": 85,
    "apnews.com": 85,
    "npr.org": 80,
    "theguardian.com": 75,
    "nytimes.com": 75,
    "washingtonpost.com": 75,
    "wsj.com": 75,
    
    # Fact-checking sites
    "snopes.com": 80,
    "factcheck.org": 85,
    "politifact.com": 80,
    "fullfact.org": 80,
    
    # Reference
    "wikipedia.org": 70,
    "britannica.com": 80,
}

# Red flag patterns for unreliable sources
RED_FLAG_PATTERNS = [
    "clickbait", "shocking", "you won't believe", "doctors hate this",
    "secret", "conspiracy", "they don't want you to know",
    "miracle cure", "instant", "guaranteed", "100% effective"
]

# Credibility indicators
CREDIBILITY_INDICATORS = {
    "positive": [
        "peer reviewed", "peer-reviewed", "study", "research", "data",
        "methodology", "references", "bibliography", "doi:",
        "published in", "journal", "university", "institute"
    ],
    "negative": [
        "opinion", "rumor", "unverified", "alleged", "claims without evidence",
        "conspiracy", "hoax", "fake", "misleading"
    ]
}


def _calculate_domain_trust_score(url: str) -> float:
    """Calculate trust score based on domain."""
    url_lower = url.lower()
    
    # Check exact domain matches
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url_lower:
            return score
    
    # Check for general patterns
    if ".edu" in url_lower:
        return 85
    elif ".gov" in url_lower:
        return 90
    elif ".org" in url_lower:
        return 65
    elif any(pattern in url_lower for pattern in ["blog", "wordpress", "medium.com"]):
        return 40
    elif any(pattern in url_lower for pattern in ["twitter", "facebook", "instagram"]):
        return 25
    else:
        return 50  # Default score


def _analyze_content_credibility(content: str, title: str = "") -> Dict[str, Any]:
    """Analyze content for credibility indicators."""
    content_lower = content.lower()
    title_lower = title.lower()
    
    analysis = {
        "credibility_score": 50,
        "positive_indicators": [],
        "negative_indicators": [],
        "red_flags": [],
        "quality_metrics": {}
    }
    
    # Check for positive indicators
    for indicator in CREDIBILITY_INDICATORS["positive"]:
        if indicator in content_lower:
            analysis["positive_indicators"].append(indicator)
            analysis["credibility_score"] += 5
    
    # Check for negative indicators
    for indicator in CREDIBILITY_INDICATORS["negative"]:
        if indicator in content_lower:
            analysis["negative_indicators"].append(indicator)
            analysis["credibility_score"] -= 10
    
    # Check for red flags
    for pattern in RED_FLAG_PATTERNS:
        if pattern in content_lower or pattern in title_lower:
            analysis["red_flags"].append(pattern)
            analysis["credibility_score"] -= 15
    
    # Quality metrics
    analysis["quality_metrics"] = {
        "content_length": len(content),
        "has_citations": bool(re.search(r'\[\d+\]|\(\d{4}\)', content)),
        "has_links": bool(re.search(r'http[s]?://', content)),
        "has_dates": bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', content)),
        "paragraph_count": len(content.split('\n\n')),
        "sentence_count": len(re.split(r'[.!?]+', content))
    }
    
    # Adjust score based on quality metrics
    if analysis["quality_metrics"]["content_length"] > 1000:
        analysis["credibility_score"] += 5
    if analysis["quality_metrics"]["has_citations"]:
        analysis["credibility_score"] += 10
    if analysis["quality_metrics"]["has_dates"]:
        analysis["credibility_score"] += 3
    
    # Ensure score is within bounds
    analysis["credibility_score"] = max(0, min(100, analysis["credibility_score"]))
    
    return analysis


def _extract_factual_claims(text: str) -> List[str]:
    """Extract factual claims from text."""
    claims = []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Too short
            continue
        
        # Look for factual statement patterns
        factual_patterns = [
            r'according to.*',
            r'research shows.*',
            r'study found.*',
            r'data indicates.*',
            r'statistics show.*',
            r'evidence suggests.*',
            r'report states.*',
            r'\d+%.*',
            r'\d+,\d+.*',
            r'in \d{4}.*'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence.lower()):
                claims.append(sentence)
                break
    
    return claims[:10]  # Limit to top 10 claims


def _verify_claim_against_sources(claim: str, sources: List[str]) -> Dict[str, Any]:
    """Verify a claim against multiple sources."""
    verification = {
        "claim": claim,
        "verification_status": "insufficient_data",
        "confidence_score": 0,
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "neutral_mentions": []
    }
    
    claim_keywords = set(claim.lower().split())
    
    for source in sources:
        source_lower = source.lower()
        source_words = set(source_lower.split())
        
        # Calculate keyword overlap
        overlap = len(claim_keywords.intersection(source_words)) / len(claim_keywords)
        
        if overlap > 0.3:  # Significant overlap
            # Check for supporting language
            if any(word in source_lower for word in ["confirms", "supports", "validates", "proves", "shows", "demonstrates"]):
                verification["supporting_evidence"].append({
                    "content": source[:200] + "..." if len(source) > 200 else source,
                    "overlap_score": overlap
                })
            # Check for contradicting language
            elif any(word in source_lower for word in ["contradicts", "disputes", "refutes", "false", "incorrect", "wrong"]):
                verification["contradicting_evidence"].append({
                    "content": source[:200] + "..." if len(source) > 200 else source,
                    "overlap_score": overlap
                })
            # Neutral mention
            elif overlap > 0.5:
                verification["neutral_mentions"].append({
                    "content": source[:200] + "..." if len(source) > 200 else source,
                    "overlap_score": overlap
                })
    
    # Determine verification status
    support_count = len(verification["supporting_evidence"])
    contradict_count = len(verification["contradicting_evidence"])
    neutral_count = len(verification["neutral_mentions"])
    
    if support_count > contradict_count and support_count > 0:
        verification["verification_status"] = "verified"
        verification["confidence_score"] = min(90, support_count * 30)
    elif contradict_count > support_count and contradict_count > 0:
        verification["verification_status"] = "disputed"
        verification["confidence_score"] = min(90, contradict_count * 30)
    elif support_count > 0 or neutral_count > 0:
        verification["verification_status"] = "partially_verified"
        verification["confidence_score"] = min(70, (support_count + neutral_count) * 20)
    else:
        verification["verification_status"] = "insufficient_data"
        verification["confidence_score"] = 10
    
    return verification


@tool(args_schema=FactCheckInput)
def fact_check_claim(claim: str, sources: List[str], context: str = "") -> str:
    """
    Fact-check a specific claim against multiple sources.
    
    Args:
        claim: The claim to fact-check
        sources: List of source content to verify against
        context: Additional context for the claim
    
    Returns:
        JSON string with fact-check results
    """
    try:
        # Create cache key
        cache_key = hashlib.md5(f"{claim}{len(sources)}".encode()).hexdigest()
        
        # Check cache
        if cache_key in _fact_check_cache:
            return json.dumps(_fact_check_cache[cache_key])
        
        # Perform fact-checking
        verification = _verify_claim_against_sources(claim, sources)
        
        # Add context analysis if provided
        if context:
            context_analysis = _analyze_content_credibility(context)
            verification["context_analysis"] = context_analysis
        
        # Extract related claims from sources
        all_source_text = " ".join(sources)
        related_claims = _extract_factual_claims(all_source_text)
        verification["related_claims"] = related_claims[:5]  # Top 5
        
        result = {
            "status": "success",
            "fact_check": verification,
            "timestamp": datetime.now().isoformat(),
            "sources_analyzed": len(sources),
            "message": f"Fact-check completed for claim: {claim[:50]}..."
        }
        
        # Cache result
        _fact_check_cache[cache_key] = result
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Fact-checking failed: {str(e)}"
        })


@tool(args_schema=CrossVerifyInput)
def cross_verify_claims(
    claims: List[str], 
    source_urls: List[str], 
    verification_threshold: float = 0.7
) -> str:
    """
    Cross-verify multiple claims against multiple sources.
    
    Args:
        claims: List of claims to verify
        source_urls: List of source URLs (content should be fetched separately)
        verification_threshold: Minimum confidence threshold for verification
    
    Returns:
        JSON string with cross-verification results
    """
    try:
        verification_results = []
        
        for claim in claims:
            # For this demo, we'll use the URLs as placeholder content
            # In practice, you'd fetch the actual content from these URLs
            placeholder_sources = [f"Content from {url}" for url in source_urls]
            
            verification = _verify_claim_against_sources(claim, placeholder_sources)
            
            # Add source credibility scores
            source_scores = []
            for url in source_urls:
                domain_score = _calculate_domain_trust_score(url)
                source_scores.append({
                    "url": url,
                    "trust_score": domain_score
                })
            
            verification["source_credibility"] = source_scores
            verification["meets_threshold"] = verification["confidence_score"] >= verification_threshold * 100
            
            verification_results.append(verification)
        
        # Calculate overall verification summary
        verified_count = sum(1 for v in verification_results if v["verification_status"] == "verified")
        disputed_count = sum(1 for v in verification_results if v["verification_status"] == "disputed")
        insufficient_count = sum(1 for v in verification_results if v["verification_status"] == "insufficient_data")
        
        avg_confidence = sum(v["confidence_score"] for v in verification_results) / len(verification_results) if verification_results else 0
        
        result = {
            "status": "success",
            "cross_verification": {
                "total_claims": len(claims),
                "verified_claims": verified_count,
                "disputed_claims": disputed_count,
                "insufficient_data_claims": insufficient_count,
                "average_confidence": avg_confidence,
                "verification_threshold": verification_threshold,
                "overall_reliability": "high" if avg_confidence >= 80 else "medium" if avg_confidence >= 60 else "low"
            },
            "detailed_results": verification_results,
            "timestamp": datetime.now().isoformat(),
            "message": f"Cross-verified {len(claims)} claims against {len(source_urls)} sources"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Cross-verification failed: {str(e)}"
        })


@tool(args_schema=SourceCredibilityInput)
def evaluate_source_credibility(url: str, content: str = "", title: str = "") -> str:
    """
    Evaluate the credibility of a source based on URL, content, and title.
    
    Args:
        url: URL of the source
        content: Content from the source
        title: Title of the source
    
    Returns:
        JSON string with credibility evaluation
    """
    try:
        # Calculate domain trust score
        domain_score = _calculate_domain_trust_score(url)
        
        # Analyze content credibility
        content_analysis = _analyze_content_credibility(content, title)
        
        # Combine scores
        final_score = (domain_score * 0.6) + (content_analysis["credibility_score"] * 0.4)
        
        # Determine credibility level
        if final_score >= 85:
            credibility_level = "very_high"
        elif final_score >= 70:
            credibility_level = "high"
        elif final_score >= 55:
            credibility_level = "medium"
        elif final_score >= 40:
            credibility_level = "low"
        else:
            credibility_level = "very_low"
        
        # Store in cache
        _credibility_scores[url] = final_score
        
        result = {
            "status": "success",
            "credibility_evaluation": {
                "url": url,
                "domain_trust_score": domain_score,
                "content_credibility_score": content_analysis["credibility_score"],
                "final_credibility_score": final_score,
                "credibility_level": credibility_level,
                "positive_indicators": content_analysis["positive_indicators"],
                "negative_indicators": content_analysis["negative_indicators"],
                "red_flags": content_analysis["red_flags"],
                "quality_metrics": content_analysis["quality_metrics"]
            },
            "recommendations": _generate_credibility_recommendations(final_score, content_analysis),
            "timestamp": datetime.now().isoformat(),
            "message": f"Credibility evaluation completed for {url}"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Credibility evaluation failed: {str(e)}"
        })


def _generate_credibility_recommendations(score: float, analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on credibility analysis."""
    recommendations = []
    
    if score < 50:
        recommendations.append("Use caution - source has low credibility score")
        recommendations.append("Verify information with additional high-trust sources")
    
    if analysis["red_flags"]:
        recommendations.append("Red flags detected - be especially cautious")
    
    if analysis["negative_indicators"]:
        recommendations.append("Negative credibility indicators found")
    
    if not analysis["positive_indicators"]:
        recommendations.append("Consider finding sources with stronger credibility indicators")
    
    if analysis["quality_metrics"]["content_length"] < 500:
        recommendations.append("Source content is quite brief - seek more comprehensive sources")
    
    if score >= 80:
        recommendations.append("High credibility source - suitable for fact-checking")
    
    return recommendations


@tool
def get_fact_check_summary() -> str:
    """
    Get summary of all fact-checking activities.
    
    Returns:
        JSON string with fact-checking summary
    """
    try:
        total_checks = len(_fact_check_cache)
        
        if total_checks == 0:
            return json.dumps({
                "status": "success",
                "summary": {
                    "total_fact_checks": 0,
                    "message": "No fact-checks performed yet"
                }
            })
        
        # Analyze cached results
        verified_count = 0
        disputed_count = 0
        insufficient_count = 0
        
        for result in _fact_check_cache.values():
            if result.get("fact_check", {}).get("verification_status") == "verified":
                verified_count += 1
            elif result.get("fact_check", {}).get("verification_status") == "disputed":
                disputed_count += 1
            else:
                insufficient_count += 1
        
        # Credibility scores summary
        avg_credibility = sum(_credibility_scores.values()) / len(_credibility_scores) if _credibility_scores else 0
        high_credibility_sources = len([s for s in _credibility_scores.values() if s >= 80])
        
        result = {
            "status": "success",
            "summary": {
                "total_fact_checks": total_checks,
                "verified_claims": verified_count,
                "disputed_claims": disputed_count,
                "insufficient_data_claims": insufficient_count,
                "verification_rate": (verified_count / total_checks * 100) if total_checks > 0 else 0,
                "sources_evaluated": len(_credibility_scores),
                "average_source_credibility": avg_credibility,
                "high_credibility_sources": high_credibility_sources
            },
            "message": f"Fact-checking summary: {total_checks} checks performed"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get fact-check summary: {str(e)}"
        })


@tool
def clear_fact_check_cache() -> str:
    """
    Clear the fact-checking cache.
    
    Returns:
        JSON string with cleanup results
    """
    try:
        global _fact_check_cache, _credibility_scores
        
        cache_count = len(_fact_check_cache)
        credibility_count = len(_credibility_scores)
        
        _fact_check_cache.clear()
        _credibility_scores.clear()
        
        return json.dumps({
            "status": "success",
            "cleared_items": {
                "fact_checks": cache_count,
                "credibility_scores": credibility_count
            },
            "message": f"Cleared {cache_count} fact-checks and {credibility_count} credibility scores"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to clear cache: {str(e)}"
        })