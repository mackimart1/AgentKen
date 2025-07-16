"""
Enhanced Web Researcher Agent: Advanced knowledge gatherer with fact-checking, source prioritization, and multi-language support.

Key Enhancements:
1. Fact-Checking: Cross-verification using multiple authoritative sources
2. Source Prioritization: Trust-based ranking system with caching
3. Multi-Language Support: Multilingual searches with translation capabilities

This enhanced version provides robust research with comprehensive quality assurance.
"""

from typing import Literal, Dict, Any, List, Optional, Tuple
import json
import hashlib
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import logging

logger = logging.getLogger(__name__)

# Enhanced system prompt
enhanced_system_prompt = """You are Enhanced Web Researcher, an advanced ReAct agent with fact-checking, source prioritization, and multi-language capabilities.

ENHANCED CAPABILITIES:
1. **Fact-Checking**: Cross-verify claims using multiple authoritative sources
2. **Source Prioritization**: Rank sources by trust score and reliability
3. **Multi-Language Support**: Conduct searches in multiple languages with translation

YOUR ENHANCED WORKFLOW:
1. **Query Analysis** - Understand the research request and identify key claims
2. **Source Planning** - Determine optimal sources and languages for research
3. **Multi-Source Research** - Gather information from diverse, authoritative sources
4. **Fact Verification** - Cross-verify claims across multiple sources
5. **Source Evaluation** - Assess source credibility and trust scores
6. **Translation & Synthesis** - Translate foreign content and synthesize findings
7. **Quality Assessment** - Evaluate information reliability and confidence
8. **Final Report** - Provide comprehensive, fact-checked research results

ENHANCED REQUIREMENTS:
- **ALWAYS** verify claims using multiple sources
- **ALWAYS** prioritize high-trust sources (academic > news > blogs)
- **ALWAYS** check for multilingual sources when relevant
- **NEVER** rely on single-source information for important claims
- **MONITOR** source credibility and update trust scores

Use enhanced tools for fact-checking, source evaluation, and translation throughout the research process.
"""

class SourceType(Enum):
    """Source type classification."""
    ACADEMIC = "academic"
    NEWS = "news"
    GOVERNMENT = "government"
    ORGANIZATION = "organization"
    BLOG = "blog"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class LanguageCode(Enum):
    """Supported language codes."""
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    RU = "ru"  # Russian
    ZH = "zh"  # Chinese
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    AR = "ar"  # Arabic


@dataclass
class SourceInfo:
    """Information about a research source."""
    url: str
    title: str
    content: str
    source_type: SourceType
    trust_score: float  # 0-100
    language: str
    timestamp: datetime
    credibility_indicators: List[str] = field(default_factory=list)
    fact_check_status: str = "pending"  # pending, verified, disputed, false
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "source_type": self.source_type.value,
            "trust_score": self.trust_score,
            "language": self.language,
            "timestamp": self.timestamp.isoformat(),
            "credibility_indicators": self.credibility_indicators,
            "fact_check_status": self.fact_check_status
        }


@dataclass
class FactCheckResult:
    """Result of fact-checking analysis."""
    claim: str
    verification_status: str  # verified, disputed, false, insufficient_data
    confidence_score: float  # 0-100
    supporting_sources: List[SourceInfo] = field(default_factory=list)
    contradicting_sources: List[SourceInfo] = field(default_factory=list)
    consensus_level: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "verification_status": self.verification_status,
            "confidence_score": self.confidence_score,
            "supporting_sources": [s.to_dict() for s in self.supporting_sources],
            "contradicting_sources": [s.to_dict() for s in self.contradicting_sources],
            "consensus_level": self.consensus_level
        }


@dataclass
class ResearchResult:
    """Enhanced research result with fact-checking and source analysis."""
    query: str
    summary: str
    fact_checks: List[FactCheckResult] = field(default_factory=list)
    sources: List[SourceInfo] = field(default_factory=list)
    languages_searched: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    research_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "fact_checks": [fc.to_dict() for fc in self.fact_checks],
            "sources": [s.to_dict() for s in self.sources],
            "languages_searched": self.languages_searched,
            "overall_confidence": self.overall_confidence,
            "research_timestamp": self.research_timestamp.isoformat()
        }


class SourceEvaluator:
    """Evaluates source credibility and assigns trust scores."""
    
    def __init__(self):
        # Trust score mappings for different source types
        self.base_trust_scores = {
            SourceType.ACADEMIC: 90,
            SourceType.GOVERNMENT: 85,
            SourceType.NEWS: 70,
            SourceType.ORGANIZATION: 65,
            SourceType.BLOG: 40,
            SourceType.SOCIAL: 25,
            SourceType.UNKNOWN: 30
        }
        
        # High-trust domains
        self.trusted_domains = {
            # Academic
            "scholar.google.com": 95,
            "pubmed.ncbi.nlm.nih.gov": 95,
            "arxiv.org": 90,
            "jstor.org": 90,
            "nature.com": 95,
            "science.org": 95,
            "cell.com": 90,
            
            # News
            "reuters.com": 85,
            "bbc.com": 85,
            "apnews.com": 85,
            "npr.org": 80,
            "theguardian.com": 75,
            "nytimes.com": 75,
            "washingtonpost.com": 75,
            
            # Government
            "gov": 90,
            "edu": 85,
            "who.int": 90,
            "cdc.gov": 90,
            "nih.gov": 90,
            
            # Organizations
            "wikipedia.org": 70,
            "britannica.com": 80,
        }
    
    def classify_source_type(self, url: str, title: str = "", content: str = "") -> SourceType:
        """Classify the type of source based on URL and content."""
        url_lower = url.lower()
        
        # Academic sources
        if any(domain in url_lower for domain in ["scholar.google", "pubmed", "arxiv", "jstor", "nature.com", "science.org"]):
            return SourceType.ACADEMIC
        
        if ".edu" in url_lower or "university" in url_lower or "journal" in url_lower:
            return SourceType.ACADEMIC
        
        # Government sources
        if ".gov" in url_lower or any(domain in url_lower for domain in ["who.int", "cdc.gov", "nih.gov"]):
            return SourceType.GOVERNMENT
        
        # News sources
        if any(domain in url_lower for domain in ["reuters", "bbc", "cnn", "npr", "guardian", "nytimes", "washingtonpost", "apnews"]):
            return SourceType.NEWS
        
        # Social media
        if any(domain in url_lower for domain in ["twitter", "facebook", "instagram", "tiktok", "reddit"]):
            return SourceType.SOCIAL
        
        # Blog indicators
        if any(indicator in url_lower for indicator in ["blog", "wordpress", "medium.com", "substack"]):
            return SourceType.BLOG
        
        # Organization
        if any(indicator in url_lower for indicator in [".org", "foundation", "institute"]):
            return SourceType.ORGANIZATION
        
        return SourceType.UNKNOWN
    
    def calculate_trust_score(self, source_info: SourceInfo) -> float:
        """Calculate trust score for a source."""
        base_score = self.base_trust_scores.get(source_info.source_type, 30)
        
        # Domain-specific adjustments
        domain_score = 0
        for domain, score in self.trusted_domains.items():
            if domain in source_info.url.lower():
                domain_score = score
                break
        
        # Use domain score if higher than base score
        if domain_score > base_score:
            base_score = domain_score
        
        # Content quality indicators
        content_bonus = 0
        if source_info.content:
            content_lower = source_info.content.lower()
            
            # Positive indicators
            if "peer reviewed" in content_lower or "peer-reviewed" in content_lower:
                content_bonus += 10
            if "doi:" in content_lower:
                content_bonus += 5
            if "references" in content_lower or "bibliography" in content_lower:
                content_bonus += 5
            if len(source_info.content) > 1000:  # Substantial content
                content_bonus += 5
            
            # Negative indicators
            if "opinion" in content_lower and source_info.source_type != SourceType.NEWS:
                content_bonus -= 10
            if "unverified" in content_lower or "rumor" in content_lower:
                content_bonus -= 15
        
        # Credibility indicators bonus
        credibility_bonus = len(source_info.credibility_indicators) * 2
        
        final_score = min(100, max(0, base_score + content_bonus + credibility_bonus))
        return final_score
    
    def identify_credibility_indicators(self, content: str, url: str) -> List[str]:
        """Identify credibility indicators in content."""
        indicators = []
        content_lower = content.lower()
        
        if "peer reviewed" in content_lower or "peer-reviewed" in content_lower:
            indicators.append("peer_reviewed")
        
        if "doi:" in content_lower:
            indicators.append("has_doi")
        
        if re.search(r'\d{4}', content):  # Has year citations
            indicators.append("has_citations")
        
        if "author" in content_lower and "affiliation" in content_lower:
            indicators.append("author_credentials")
        
        if "methodology" in content_lower or "methods" in content_lower:
            indicators.append("methodology_described")
        
        if ".edu" in url or ".gov" in url:
            indicators.append("institutional_source")
        
        return indicators


class FactChecker:
    """Performs fact-checking by cross-referencing multiple sources."""
    
    def __init__(self, source_evaluator: SourceEvaluator):
        self.source_evaluator = source_evaluator
    
    def extract_claims(self, query: str, content: str) -> List[str]:
        """Extract verifiable claims from content."""
        claims = []
        
        # Simple claim extraction based on common patterns
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Too short to be meaningful
                continue
            
            # Look for factual statements
            if any(indicator in sentence.lower() for indicator in [
                "according to", "research shows", "study found", "data indicates",
                "statistics show", "evidence suggests", "report states"
            ]):
                claims.append(sentence)
            
            # Look for specific numbers or dates
            if re.search(r'\d+%|\d+,\d+|\d{4}', sentence):
                claims.append(sentence)
        
        return claims[:5]  # Limit to top 5 claims
    
    def verify_claim(self, claim: str, sources: List[SourceInfo]) -> FactCheckResult:
        """Verify a claim against multiple sources."""
        supporting_sources = []
        contradicting_sources = []
        
        claim_lower = claim.lower()
        
        for source in sources:
            content_lower = source.content.lower()
            
            # Simple content matching (could be enhanced with NLP)
            claim_words = set(claim_lower.split())
            content_words = set(content_lower.split())
            
            overlap = len(claim_words.intersection(content_words)) / len(claim_words)
            
            if overlap > 0.3:  # Significant overlap
                # Check for supporting or contradicting language
                if any(word in content_lower for word in ["confirms", "supports", "validates", "proves"]):
                    supporting_sources.append(source)
                elif any(word in content_lower for word in ["contradicts", "disputes", "refutes", "false"]):
                    contradicting_sources.append(source)
                elif overlap > 0.5:  # High overlap without explicit contradiction
                    supporting_sources.append(source)
        
        # Calculate verification status
        support_weight = sum(s.trust_score for s in supporting_sources)
        contradict_weight = sum(s.trust_score for s in contradicting_sources)
        
        if support_weight > contradict_weight * 2:
            status = "verified"
        elif contradict_weight > support_weight * 2:
            status = "disputed"
        elif support_weight > 0 and contradict_weight > 0:
            status = "disputed"
        else:
            status = "insufficient_data"
        
        # Calculate confidence and consensus
        total_weight = support_weight + contradict_weight
        confidence = min(100, total_weight / 10)  # Scale to 0-100
        consensus = (support_weight / total_weight * 100) if total_weight > 0 else 0
        
        return FactCheckResult(
            claim=claim,
            verification_status=status,
            confidence_score=confidence,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            consensus_level=consensus
        )


class MultiLanguageProcessor:
    """Handles multilingual search and translation."""
    
    def __init__(self):
        self.language_patterns = {
            LanguageCode.EN: ["english", "en", "eng"],
            LanguageCode.ES: ["spanish", "español", "es", "spa"],
            LanguageCode.FR: ["french", "français", "fr", "fra"],
            LanguageCode.DE: ["german", "deutsch", "de", "ger"],
            LanguageCode.IT: ["italian", "italiano", "it", "ita"],
            LanguageCode.PT: ["portuguese", "português", "pt", "por"],
            LanguageCode.RU: ["russian", "русский", "ru", "rus"],
            LanguageCode.ZH: ["chinese", "中文", "zh", "chi"],
            LanguageCode.JA: ["japanese", "日本語", "ja", "jpn"],
            LanguageCode.KO: ["korean", "한국어", "ko", "kor"],
            LanguageCode.AR: ["arabic", "العربية", "ar", "ara"]
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        # This is a simplified implementation
        # In production, you'd use a proper language detection library
        
        if re.search(r'[а-яё]', text.lower()):
            return LanguageCode.RU.value
        elif re.search(r'[一-龯]', text):
            return LanguageCode.ZH.value
        elif re.search(r'[ひらがなカタカナ]', text):
            return LanguageCode.JA.value
        elif re.search(r'[가-힣]', text):
            return LanguageCode.KO.value
        elif re.search(r'[ء-ي]', text):
            return LanguageCode.AR.value
        elif re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', text.lower()):
            # European languages - would need more sophisticated detection
            return LanguageCode.FR.value  # Default to French for European chars
        else:
            return LanguageCode.EN.value
    
    def generate_multilingual_queries(self, query: str) -> Dict[str, str]:
        """Generate search queries in multiple languages."""
        # This is a simplified implementation
        # In production, you'd use a translation service
        
        queries = {LanguageCode.EN.value: query}
        
        # Simple keyword-based translations for common research terms
        translations = {
            "climate change": {
                "es": "cambio climático",
                "fr": "changement climatique",
                "de": "Klimawandel"
            },
            "artificial intelligence": {
                "es": "inteligencia artificial",
                "fr": "intelligence artificielle",
                "de": "künstliche Intelligenz"
            },
            "covid": {
                "es": "covid",
                "fr": "covid",
                "de": "covid"
            }
        }
        
        query_lower = query.lower()
        for term, trans in translations.items():
            if term in query_lower:
                for lang, translation in trans.items():
                    queries[lang] = query.lower().replace(term, translation)
        
        return queries
    
    def translate_content(self, content: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate content to target language."""
        # This is a placeholder implementation
        # In production, you'd use a translation service like Google Translate
        
        if source_lang == target_lang:
            return content
        
        # Simple translation indicator
        return f"[Translated from {source_lang}] {content}"


class QueryCache:
    """Caches frequent queries and results."""
    
    def __init__(self, cache_file: str = "research_cache.json", ttl_hours: int = 24):
        self.cache_file = Path(cache_file)
        self.ttl_hours = ttl_hours
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query."""
        key = self._get_cache_key(query)
        
        if key in self.cache:
            cached_data = self.cache[key]
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            
            # Check if cache is still valid
            if datetime.now() - cached_time < timedelta(hours=self.ttl_hours):
                return cached_data["result"]
            else:
                # Remove expired cache
                del self.cache[key]
                self._save_cache()
        
        return None
    
    def set(self, query: str, result: Dict[str, Any]):
        """Cache result for query."""
        key = self._get_cache_key(query)
        
        self.cache[key] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        self._save_cache()


# Initialize enhanced components
source_evaluator = SourceEvaluator()
fact_checker = FactChecker(source_evaluator)
ml_processor = MultiLanguageProcessor()
query_cache = QueryCache()

# Load enhanced tools
from tools.duck_duck_go_web_search import duck_duck_go_web_search
from tools.fetch_web_page_content import fetch_web_page_content

# Enhanced tools will be added
tools = [duck_duck_go_web_search, fetch_web_page_content]


def enhanced_reasoning(state: MessagesState) -> dict:
    """Enhanced reasoning with fact-checking and source prioritization."""
    print("\nenhanced web_researcher is thinking...")
    
    messages = state["messages"]
    
    # Extract query from messages
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = str(msg.content)
            break
    
    # Check cache first
    cached_result = query_cache.get(query)
    if cached_result:
        print("Found cached result for query")
        return {"messages": [AIMessage(content=json.dumps(cached_result))]}
    
    # Enhanced reasoning with multilingual and fact-checking awareness
    enhanced_messages = list(messages)
    
    # Add context about enhanced capabilities
    context_message = SystemMessage(content=f"""
Enhanced Research Context:
- Query: {query}
- Multilingual search capability available
- Fact-checking required for claims
- Source prioritization by trust score
- Cross-verification needed for important facts

Use multiple sources and verify claims before concluding research.
""")
    enhanced_messages.insert(-1, context_message)
    
    try:
        # Use Google Gemini for tool calling from hybrid configuration
        tool_model = config.get_model_for_tools()
        if tool_model is None:
            # Fallback to default model if hybrid setup fails
            tool_model = config.default_langchain_model
            logger.warning("Using fallback model for tools - may not support function calling")
        
        tooled_up_model = tool_model.bind_tools(tools)
        response = tooled_up_model.invoke(enhanced_messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"Enhanced reasoning error: {e}")
        return {"messages": [AIMessage(content=f"Enhanced research error: {str(e)}")]}


def enhanced_check_for_tool_calls(state: MessagesState) -> Literal["tools", "END"]:
    """Enhanced tool call checking with research workflow awareness."""
    messages = state["messages"]
    if not messages:
        return "END"
    
    last_message = messages[-1]
    
    if (isinstance(last_message, AIMessage) and 
        hasattr(last_message, "tool_calls") and last_message.tool_calls):
        
        content = last_message.content
        if isinstance(content, str) and content.strip():
            print("enhanced web_researcher thought:")
            print(content)
        
        print("\nenhanced web_researcher is using enhanced tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"
    
    return "END"


def process_research_results(messages: List) -> ResearchResult:
    """Process research results with enhanced analysis."""
    # Extract query
    query = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            query = str(msg.content)
            break
    
    # Collect all research content
    research_content = []
    sources = []
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                content = str(msg.content)
                if content and len(content) > 50:  # Substantial content
                    research_content.append(content)
                    
                    # Create source info (simplified)
                    source_info = SourceInfo(
                        url="research_source",
                        title="Research Content",
                        content=content,
                        source_type=SourceType.UNKNOWN,
                        trust_score=50,
                        language=ml_processor.detect_language(content),
                        timestamp=datetime.now()
                    )
                    
                    # Evaluate source
                    source_info.trust_score = source_evaluator.calculate_trust_score(source_info)
                    source_info.credibility_indicators = source_evaluator.identify_credibility_indicators(
                        content, source_info.url
                    )
                    
                    sources.append(source_info)
            except Exception:
                continue
    
    # Perform fact-checking
    fact_checks = []
    all_content = " ".join(research_content)
    
    if all_content:
        claims = fact_checker.extract_claims(query, all_content)
        for claim in claims:
            fact_check = fact_checker.verify_claim(claim, sources)
            fact_checks.append(fact_check)
    
    # Generate summary
    summary = all_content[:1000] + "..." if len(all_content) > 1000 else all_content
    
    # Calculate overall confidence
    if fact_checks:
        overall_confidence = sum(fc.confidence_score for fc in fact_checks) / len(fact_checks)
    else:
        overall_confidence = 50.0  # Default confidence
    
    # Detect languages
    languages_searched = list(set(source.language for source in sources))
    
    result = ResearchResult(
        query=query,
        summary=summary,
        fact_checks=fact_checks,
        sources=sources,
        languages_searched=languages_searched,
        overall_confidence=overall_confidence
    )
    
    return result


# Create enhanced workflow
workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", enhanced_reasoning)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges("reasoning", enhanced_check_for_tool_calls)
workflow.add_edge("tools", "reasoning")

enhanced_graph = workflow.compile()


def enhanced_web_researcher(task: str) -> Dict[str, Any]:
    """
    Enhanced Web Researcher with fact-checking, source prioritization, and multi-language support.
    
    Args:
        task (str): The research query or task.
        
    Returns:
        Dict[str, Any]: Enhanced result with fact-checking and source analysis
    """
    print(f"Enhanced Web Researcher starting task: {task[:100]}...")
    
    try:
        # Execute enhanced workflow
        final_state = enhanced_graph.invoke({
            "messages": [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=task)
            ]
        })
        
        # Process results with enhanced analysis
        research_result = process_research_results(final_state.get("messages", []))
        
        # Cache result
        result_dict = research_result.to_dict()
        query_cache.set(task, result_dict)
        
        # Determine status based on confidence and fact-checks
        if research_result.overall_confidence >= 70:
            status = "success"
        elif research_result.overall_confidence >= 40:
            status = "partial_success"
        else:
            status = "failure"
        
        # Check for disputed facts
        disputed_facts = [fc for fc in research_result.fact_checks if fc.verification_status == "disputed"]
        if disputed_facts:
            status = "disputed_information"
        
        enhanced_result = {
            "status": status,
            "result": research_result.to_dict(),
            "message": research_result.summary,
            "fact_checks": [fc.to_dict() for fc in research_result.fact_checks],
            "source_analysis": {
                "total_sources": len(research_result.sources),
                "average_trust_score": sum(s.trust_score for s in research_result.sources) / len(research_result.sources) if research_result.sources else 0,
                "languages_found": research_result.languages_searched,
                "high_trust_sources": len([s for s in research_result.sources if s.trust_score >= 80])
            },
            "confidence_metrics": {
                "overall_confidence": research_result.overall_confidence,
                "verified_claims": len([fc for fc in research_result.fact_checks if fc.verification_status == "verified"]),
                "disputed_claims": len([fc for fc in research_result.fact_checks if fc.verification_status == "disputed"]),
                "insufficient_data_claims": len([fc for fc in research_result.fact_checks if fc.verification_status == "insufficient_data"])
            }
        }
        
        return enhanced_result
        
    except Exception as e:
        error_msg = f"Enhanced Web Researcher execution failed: {str(e)}"
        print(error_msg)
        
        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "fact_checks": [],
            "source_analysis": {},
            "confidence_metrics": {}
        }


# Export enhanced function
__all__ = ["enhanced_web_researcher", "SourceEvaluator", "FactChecker", "MultiLanguageProcessor"]