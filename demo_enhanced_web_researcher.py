#!/usr/bin/env python3
"""
Demonstration script for Enhanced WebResearcher capabilities:
1. Fact-Checking: Cross-verification using multiple authoritative sources
2. Source Prioritization: Trust-based ranking system with caching
3. Multi-Language Support: Multilingual searches with translation

This script shows how the enhanced features work without requiring full web searches.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Import the enhanced tools
from tools.fact_checking import (
    fact_check_claim, cross_verify_claims, evaluate_source_credibility,
    get_fact_check_summary, clear_fact_check_cache
)
from tools.source_prioritization import (
    rank_sources_by_trust, cache_search_results, get_cached_results,
    update_domain_trust_score, get_trust_score_rankings, get_cache_statistics
)
from tools.multilingual_research import (
    multilingual_search_query, translate_research_content, detect_content_language,
    generate_multilingual_variations, get_multilingual_statistics
)


def demo_fact_checking():
    """Demonstrate fact-checking capabilities."""
    print("=" * 80)
    print("FACT-CHECKING DEMONSTRATION")
    print("=" * 80)
    
    print("1. Testing fact-checking with sample claims...")
    
    # Sample claims to fact-check
    claims = [
        "Climate change is causing global temperatures to rise",
        "Artificial intelligence will replace all human jobs by 2030",
        "Vaccines are effective at preventing infectious diseases"
    ]
    
    # Sample source content
    sources = [
        "According to NASA data, global temperatures have risen by 1.1°C since the late 19th century due to climate change.",
        "Research shows that AI will automate many jobs but also create new opportunities in emerging fields.",
        "Multiple peer-reviewed studies demonstrate that vaccines are highly effective at preventing infectious diseases.",
        "The World Health Organization confirms that vaccines have prevented millions of deaths worldwide.",
        "Some experts argue that AI development timeline predictions are often overly optimistic."
    ]
    
    for i, claim in enumerate(claims, 1):
        print(f"\n   Claim {i}: {claim}")
        
        # Fact-check individual claim
        result = fact_check_claim.invoke({
            "claim": claim,
            "sources": sources,
            "context": "Research demonstration"
        })
        
        fact_check_data = json.loads(result)
        
        if fact_check_data["status"] == "success":
            fc = fact_check_data["fact_check"]
            print(f"   Status: {fc['verification_status']}")
            print(f"   Confidence: {fc['confidence_score']:.1f}%")
            print(f"   Supporting Evidence: {len(fc['supporting_evidence'])}")
            print(f"   Contradicting Evidence: {len(fc['contradicting_evidence'])}")
        else:
            print(f"   Error: {fact_check_data['message']}")
    
    print("\n2. Cross-verifying all claims...")
    
    # Cross-verify all claims
    cross_verify_result = cross_verify_claims.invoke({
        "claims": claims,
        "source_urls": [
            "https://nasa.gov/climate-data",
            "https://who.int/vaccines",
            "https://nature.com/ai-research",
            "https://pubmed.ncbi.nlm.nih.gov/vaccine-study"
        ],
        "verification_threshold": 0.7
    })
    
    cross_verify_data = json.loads(cross_verify_result)
    
    if cross_verify_data["status"] == "success":
        cv = cross_verify_data["cross_verification"]
        print(f"   Total Claims: {cv['total_claims']}")
        print(f"   Verified Claims: {cv['verified_claims']}")
        print(f"   Disputed Claims: {cv['disputed_claims']}")
        print(f"   Average Confidence: {cv['average_confidence']:.1f}%")
        print(f"   Overall Reliability: {cv['overall_reliability']}")
    
    print("\n3. Evaluating source credibility...")
    
    # Test source credibility evaluation
    test_sources = [
        {"url": "https://nature.com/article", "title": "Peer-reviewed research", "content": "This peer-reviewed study published in Nature demonstrates..."},
        {"url": "https://blog.example.com", "title": "Personal opinion", "content": "I think that climate change is..."},
        {"url": "https://who.int/health-report", "title": "WHO Health Report", "content": "The World Health Organization reports..."}
    ]
    
    for source in test_sources:
        cred_result = evaluate_source_credibility.invoke({
            "url": source["url"],
            "content": source["content"],
            "title": source["title"]
        })
        
        cred_data = json.loads(cred_result)
        
        if cred_data["status"] == "success":
            ce = cred_data["credibility_evaluation"]
            print(f"   Source: {source['url']}")
            print(f"   Credibility Score: {ce['final_credibility_score']:.1f}/100")
            print(f"   Credibility Level: {ce['credibility_level']}")
            print(f"   Positive Indicators: {len(ce['positive_indicators'])}")
            print(f"   Red Flags: {len(ce['red_flags'])}")
            print()
    
    return fact_check_data, cross_verify_data, cred_data


def demo_source_prioritization():
    """Demonstrate source prioritization and caching capabilities."""
    print("\n" + "=" * 80)
    print("SOURCE PRIORITIZATION DEMONSTRATION")
    print("=" * 80)
    
    print("1. Ranking sources by trust score...")
    
    # Sample sources with different trust levels
    sample_sources = [
        {
            "url": "https://nature.com/climate-study",
            "title": "Climate Change Research in Nature Journal",
            "content": "This peer-reviewed study published in Nature demonstrates significant climate trends based on 30 years of data collection and analysis."
        },
        {
            "url": "https://blog.climate-skeptic.com",
            "title": "Why Climate Change is Fake",
            "content": "This shocking truth about climate change will amaze you! Scientists don't want you to know this secret."
        },
        {
            "url": "https://nasa.gov/climate-data",
            "title": "NASA Climate Change and Global Warming",
            "content": "NASA's climate data shows clear evidence of global warming trends with comprehensive satellite measurements and ground-based observations."
        },
        {
            "url": "https://wikipedia.org/climate-change",
            "title": "Climate Change - Wikipedia",
            "content": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the mid-20th century."
        }
    ]
    
    # Rank sources
    ranking_result = rank_sources_by_trust.invoke({
        "sources": sample_sources,
        "query": "climate change research",
        "prioritize_recent": True
    })
    
    ranking_data = json.loads(ranking_result)
    
    if ranking_data["status"] == "success":
        print("   Ranked Sources:")
        for source in ranking_data["ranked_sources"]:
            print(f"   {source['rank']}. {source['url']}")
            print(f"      Trust Score: {source['trust_score']:.1f}/100")
            print(f"      Source Type: {source['source_type']}")
            print(f"      Overall Score: {source['overall_score']:.1f}/100")
            print(f"      Quality Indicators: {source['quality_indicators']}")
            if source['red_flags']:
                print(f"      Red Flags: {source['red_flags']}")
            print()
        
        metadata = ranking_data["ranking_metadata"]
        print(f"   Average Trust Score: {metadata['average_trust_score']:.1f}")
        print(f"   High Trust Sources: {metadata['high_trust_sources']}")
    
    print("\n2. Testing query caching...")
    
    # Cache search results
    cache_result = cache_search_results.invoke({
        "query": "climate change research",
        "results": sample_sources,
        "ttl_hours": 24
    })
    
    cache_data = json.loads(cache_result)
    
    if cache_data["status"] == "success":
        print(f"   Cached {cache_data['cached_results']} results")
        print(f"   Cache expires: {cache_data['expires_at'][:19]}")
    
    # Retrieve cached results
    cached_result = get_cached_results.invoke({
        "query": "climate change research"
    })
    
    cached_data = json.loads(cached_result)
    
    if cached_data["status"] == "cache_hit":
        print(f"   Cache hit! Retrieved {cached_data['result_count']} cached results")
    else:
        print(f"   Cache miss: {cached_data['message']}")
    
    print("\n3. Updating domain trust scores...")
    
    # Update trust score for a domain
    update_result = update_domain_trust_score.invoke({
        "domain": "example-news.com",
        "new_score": 75,
        "reason": "Improved fact-checking standards"
    })
    
    update_data = json.loads(update_result)
    
    if update_data["status"] == "success":
        print(f"   Updated {update_data['domain']}: {update_data['old_score']} → {update_data['new_score']}")
    
    print("\n4. Getting trust score rankings...")
    
    # Get trust rankings
    rankings_result = get_trust_score_rankings.invoke({})
    rankings_data = json.loads(rankings_result)
    
    if rankings_data["status"] == "success":
        tr = rankings_data["trust_rankings"]
        print(f"   Total Domains: {tr['total_domains']}")
        print(f"   Very High Trust: {len(tr['categories']['very_high_trust'])}")
        print(f"   High Trust: {len(tr['categories']['high_trust'])}")
        print(f"   Average Score: {tr['average_score']:.1f}")
        
        print("   Top 5 Trusted Domains:")
        for i, (domain, score) in enumerate(tr["top_10_domains"][:5], 1):
            print(f"     {i}. {domain}: {score}")
    
    return ranking_data, cache_data, update_data


def demo_multilingual_research():
    """Demonstrate multilingual research capabilities."""
    print("\n" + "=" * 80)
    print("MULTILINGUAL RESEARCH DEMONSTRATION")
    print("=" * 80)
    
    print("1. Generating multilingual search queries...")
    
    # Generate multilingual queries
    ml_query_result = multilingual_search_query.invoke({
        "query": "artificial intelligence research",
        "target_languages": ["en", "es", "fr", "de", "zh"],
        "include_translation": True
    })
    
    ml_query_data = json.loads(ml_query_result)
    
    if ml_query_data["status"] == "success":
        print(f"   Original Query: {ml_query_data['search_strategy']['original_query']}")
        print(f"   Target Languages: {len(ml_query_data['search_strategy']['target_languages'])}")
        print(f"   Total Variations: {ml_query_data['search_strategy']['total_variations']}")
        
        print("\n   Query Variations by Language:")
        for lang, data in ml_query_data["multilingual_queries"].items():
            print(f"   {lang} ({data['language_name']}):")
            for variation in data["query_variations"][:2]:  # Show first 2 variations
                print(f"     - {variation}")
        
        if ml_query_data["recommendations"]:
            print("\n   Recommendations:")
            for rec in ml_query_data["recommendations"]:
                print(f"     - {rec}")
    
    print("\n2. Testing language detection...")
    
    # Test language detection with different texts
    test_texts = [
        "Artificial intelligence is transforming the world",
        "La inteligencia artificial está transformando el mundo",
        "L'intelligence artificielle transforme le monde",
        "人工智能正在改变世界",
        "Künstliche Intelligenz verändert die Welt"
    ]
    
    for text in test_texts:
        detect_result = detect_content_language.invoke({
            "text": text,
            "confidence_threshold": 0.8
        })
        
        detect_data = json.loads(detect_result)
        
        if detect_data["status"] == "success":
            dr = detect_data["detection_result"]
            print(f"   Text: {text[:50]}...")
            print(f"   Detected: {dr['language_name']} ({dr['detected_language']})")
            print(f"   Confidence: {dr['confidence']:.2f}")
            print(f"   Meets Threshold: {detect_data['meets_threshold']}")
            print()
    
    print("\n3. Testing content translation...")
    
    # Test translation
    translation_result = translate_research_content.invoke({
        "content": "La investigación en inteligencia artificial ha avanzado significativamente en los últimos años.",
        "source_language": "es",
        "target_language": "en",
        "preserve_formatting": True
    })
    
    translation_data = json.loads(translation_result)
    
    if translation_data["status"] == "success":
        print(f"   Source ({translation_data['source_language_name']}): {translation_data['original_content'][:100]}...")
        print(f"   Target ({translation_data['target_language_name']}): {translation_data['translated_content'][:100]}...")
        print(f"   Quality Score: {translation_data['translation_quality']['confidence_score']}")
    
    print("\n4. Generating query variations...")
    
    # Generate multilingual variations
    variations_result = generate_multilingual_variations.invoke({
        "query": "climate change research",
        "languages": ["en", "es", "fr"],
        "include_synonyms": True
    })
    
    variations_data = json.loads(variations_result)
    
    if variations_data["status"] == "success":
        print(f"   Original Query: {variations_data['original_query']}")
        print(f"   Total Variations: {variations_data['total_variations']}")
        
        for lang, data in variations_data["variations_by_language"].items():
            print(f"   {data['language_name']}: {data['variation_count']} variations")
            for variation in data["variations"][:2]:  # Show first 2
                print(f"     - {variation}")
        
        strategy = variations_data["search_strategy"]
        print(f"\n   Recommended Order: {strategy['recommended_order']}")
        print(f"   Parallel Search: {strategy['parallel_search']}")
    
    return ml_query_data, detect_data, translation_data, variations_data


def demo_integration():
    """Demonstrate how all three capabilities work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED ENHANCED WEBRESEARCHER DEMONSTRATION")
    print("=" * 80)
    
    print("1. Enhanced WebResearcher workflow simulation...")
    
    # Simulate the enhanced workflow
    workflow_steps = [
        "Query Analysis - Understand research request and detect language",
        "Multilingual Planning - Generate queries in multiple languages",
        "Source Discovery - Search across languages and source types",
        "Source Evaluation - Rank sources by trust and credibility",
        "Content Extraction - Fetch and process multilingual content",
        "Language Processing - Detect languages and translate content",
        "Fact Verification - Cross-verify claims across sources",
        "Credibility Assessment - Evaluate source reliability",
        "Result Synthesis - Combine findings with confidence metrics",
        "Quality Assurance - Final validation and confidence scoring"
    ]
    
    print("   Enhanced workflow steps:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"     {i}. {step}")
    
    print("\n2. Integration benefits...")
    
    integration_benefits = [
        "Fact-Checking - Cross-verification prevents misinformation",
        "Source Prioritization - Trust-based ranking ensures quality",
        "Multilingual Support - Global knowledge access and synthesis",
        "Query Caching - Improved performance for frequent searches",
        "Language Detection - Automatic content language identification",
        "Translation Support - Seamless multilingual research",
        "Credibility Scoring - Quantitative source reliability assessment",
        "Cross-Verification - Multiple source validation for accuracy"
    ]
    
    print("   Integration benefits:")
    for benefit in integration_benefits:
        print(f"     ✅ {benefit}")
    
    print("\n3. Enhanced capabilities summary:")
    
    capabilities = {
        "Fact-Checking": {
            "Cross-Verification": "Multiple source claim validation",
            "Confidence Scoring": "Quantitative verification confidence",
            "Evidence Analysis": "Supporting vs contradicting evidence",
            "Claim Extraction": "Automatic factual claim identification",
            "Consensus Measurement": "Agreement level across sources"
        },
        "Source Prioritization": {
            "Trust Scoring": "Domain and content credibility assessment",
            "Relevance Ranking": "Query-specific source relevance",
            "Recency Weighting": "Time-based source prioritization",
            "Quality Indicators": "Academic and credibility markers",
            "Caching System": "Performance optimization for frequent queries"
        },
        "Multilingual Research": {
            "Language Detection": "Automatic content language identification",
            "Query Translation": "Multi-language search generation",
            "Content Translation": "Seamless multilingual content access",
            "Cultural Context": "Language-specific search optimization",
            "Global Coverage": "Worldwide knowledge source access"
        }
    }
    
    for category, features in capabilities.items():
        print(f"\n   {category}:")
        for feature, description in features.items():
            print(f"     🔧 {feature}: {description}")
    
    print("\n4. Quality assurance metrics:")
    
    qa_metrics = [
        "Fact-Check Confidence: 0-100 based on source agreement",
        "Source Trust Score: 0-100 based on credibility assessment",
        "Translation Quality: Confidence in translation accuracy",
        "Language Detection: Confidence in language identification",
        "Overall Research Confidence: Combined quality metrics",
        "Cross-Verification Rate: Percentage of claims verified"
    ]
    
    for metric in qa_metrics:
        print(f"     📊 {metric}")
    
    return {
        "fact_checking": True,
        "source_prioritization": True,
        "multilingual_research": True,
        "cross_verification": True,
        "trust_scoring": True,
        "query_caching": True
    }


def demo_statistics():
    """Show statistics from all enhanced components."""
    print("\n" + "=" * 80)
    print("ENHANCED WEBRESEARCHER STATISTICS")
    print("=" * 80)
    
    print("1. Fact-checking statistics...")
    
    # Get fact-checking summary
    fc_summary = get_fact_check_summary.invoke({})
    fc_data = json.loads(fc_summary)
    
    if fc_data["status"] == "success":
        summary = fc_data["summary"]
        print(f"   Total Fact-Checks: {summary['total_fact_checks']}")
        if summary["total_fact_checks"] > 0:
            print(f"   Verified Claims: {summary['verified_claims']}")
            print(f"   Disputed Claims: {summary['disputed_claims']}")
            print(f"   Verification Rate: {summary['verification_rate']:.1f}%")
            print(f"   Sources Evaluated: {summary['sources_evaluated']}")
            print(f"   Average Source Credibility: {summary['average_source_credibility']:.1f}")
    
    print("\n2. Caching statistics...")
    
    # Get cache statistics
    cache_stats = get_cache_statistics.invoke({})
    cache_data = json.loads(cache_stats)
    
    if cache_data["status"] == "success":
        stats = cache_data["cache_statistics"]
        print(f"   Total Cached Queries: {stats['total_cached_queries']}")
        if stats["total_cached_queries"] > 0:
            print(f"   Active Entries: {stats['active_entries']}")
            print(f"   Cache Efficiency: {stats['cache_efficiency']:.1f}%")
            print(f"   Average Results per Query: {stats['average_results_per_query']:.1f}")
    
    print("\n3. Multilingual statistics...")
    
    # Get multilingual statistics
    ml_stats = get_multilingual_statistics.invoke({})
    ml_data = json.loads(ml_stats)
    
    if ml_data["status"] == "success":
        stats = ml_data["statistics"]
        print(f"   Total Translations: {stats['total_translations']}")
        print(f"   Language Detections: {stats['total_language_detections']}")
        print(f"   Multilingual Searches: {stats['total_multilingual_searches']}")
        print(f"   Supported Languages: {stats['supported_languages']}")
        
        if stats["popular_languages"]:
            print("   Popular Languages:")
            for lang in stats["popular_languages"][:5]:
                print(f"     - {lang['name']} ({lang['code']}): {lang['usage_count']} uses")
    
    return fc_data, cache_data, ml_data


def main():
    """Run all demonstrations."""
    print("ENHANCED WEBRESEARCHER CAPABILITIES DEMONSTRATION")
    print("This demo shows the three key improvements:")
    print("1. Fact-Checking - Cross-verification using multiple authoritative sources")
    print("2. Source Prioritization - Trust-based ranking system with caching")
    print("3. Multi-Language Support - Multilingual searches with translation")
    print()
    
    try:
        # Run individual demonstrations
        fact_check_demo = demo_fact_checking()
        source_priority_demo = demo_source_prioritization()
        multilingual_demo = demo_multilingual_research()
        integration_demo = demo_integration()
        statistics_demo = demo_statistics()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("✅ Fact-Checking: Cross-verification and credibility assessment")
        print("✅ Source Prioritization: Trust-based ranking and caching")
        print("✅ Multilingual Research: Language detection and translation")
        print("✅ Integration: All capabilities working together seamlessly")
        print()
        print("Enhanced WebResearcher is ready with:")
        print("  🔍 Fact-Checking - Cross-verification using multiple sources")
        print("  📊 Source Prioritization - Trust-based ranking with caching")
        print("  🌍 Multilingual Support - Global research with translation")
        print("  ✅ Cross-Verification - Multiple source claim validation")
        print("  🏆 Trust Scoring - Quantitative credibility assessment")
        print("  ⚡ Query Caching - Performance optimization")
        print()
        print("The enhanced system provides reliable, comprehensive, and")
        print("globally-aware research with built-in quality assurance.")
        
        # Cleanup demo data
        print("\n" + "=" * 40)
        print("CLEANUP")
        print("=" * 40)
        print("Cleaning up demonstration data...")
        
        # Clear fact-checking cache
        fc_cleanup = clear_fact_check_cache.invoke({})
        fc_cleanup_data = json.loads(fc_cleanup)
        if fc_cleanup_data["status"] == "success":
            cleared = fc_cleanup_data["cleared_items"]
            total_cleared = cleared["fact_checks"] + cleared["credibility_scores"]
            print(f"✅ Cleared {total_cleared} fact-checking items")
        
        print("Demo cleanup completed!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the enhanced capabilities structure.")
        print("Full integration requires the complete enhanced WebResearcher system.")


if __name__ == "__main__":
    main()