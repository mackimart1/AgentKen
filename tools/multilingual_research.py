from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path


class MultilingualSearchInput(BaseModel):
    query: str = Field(description="Search query to translate and search in multiple languages")
    target_languages: List[str] = Field(default=["en", "es", "fr", "de"], description="Language codes to search in")
    include_translation: bool = Field(default=True, description="Include translation of results")


class TranslateContentInput(BaseModel):
    content: str = Field(description="Content to translate")
    source_language: str = Field(description="Source language code")
    target_language: str = Field(default="en", description="Target language code")
    preserve_formatting: bool = Field(default=True, description="Preserve original formatting")


class DetectLanguageInput(BaseModel):
    text: str = Field(description="Text to detect language for")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for detection")


class GenerateQueryVariationsInput(BaseModel):
    query: str = Field(description="Original query to generate variations for")
    languages: List[str] = Field(description="Languages to generate variations in")
    include_synonyms: bool = Field(default=True, description="Include synonym variations")


# Language mappings and patterns
LANGUAGE_CODES = {
    "en": {"name": "English", "native": "English"},
    "es": {"name": "Spanish", "native": "Español"},
    "fr": {"name": "French", "native": "Français"},
    "de": {"name": "German", "native": "Deutsch"},
    "it": {"name": "Italian", "native": "Italiano"},
    "pt": {"name": "Portuguese", "native": "Português"},
    "ru": {"name": "Russian", "native": "Русский"},
    "zh": {"name": "Chinese", "native": "中文"},
    "ja": {"name": "Japanese", "native": "日本語"},
    "ko": {"name": "Korean", "native": "한국어"},
    "ar": {"name": "Arabic", "native": "العربية"},
    "hi": {"name": "Hindi", "native": "हिन्दी"},
    "nl": {"name": "Dutch", "native": "Nederlands"},
    "sv": {"name": "Swedish", "native": "Svenska"},
    "no": {"name": "Norwegian", "native": "Norsk"},
    "da": {"name": "Danish", "native": "Dansk"},
    "fi": {"name": "Finnish", "native": "Suomi"},
    "pl": {"name": "Polish", "native": "Polski"},
    "tr": {"name": "Turkish", "native": "Türkçe"},
    "he": {"name": "Hebrew", "native": "עברית"}
}

# Character patterns for language detection
LANGUAGE_PATTERNS = {
    "en": r"[a-zA-Z\s.,!?;:'\"-]+",
    "es": r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s.,!?;:'\"-]+",
    "fr": r"[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s.,!?;:'\"-]+",
    "de": r"[a-zA-ZäöüßÄÖÜ\s.,!?;:'\"-]+",
    "it": r"[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ\s.,!?;:'\"-]+",
    "pt": r"[a-zA-ZáâãàéêíóôõúçÁÂÃÀÉÊÍÓÔÕÚÇ\s.,!?;:'\"-]+",
    "ru": r"[а-яёА-ЯЁ\s.,!?;:'\"-]+",
    "zh": r"[一-龯\s.,!?;:'\"-]+",
    "ja": r"[ひらがなカタカナ一-龯\s.,!?;:'\"-]+",
    "ko": r"[가-힣\s.,!?;:'\"-]+",
    "ar": r"[ء-ي\s.,!?;:'\"-]+",
    "hi": r"[अ-ह\s.,!?;:'\"-]+",
    "he": r"[א-ת\s.,!?;:'\"-]+"
}

# Common term translations for research queries
TERM_TRANSLATIONS = {
    "climate change": {
        "es": "cambio climático",
        "fr": "changement climatique",
        "de": "Klimawandel",
        "it": "cambiamento climatico",
        "pt": "mudança climática",
        "ru": "изменение климата",
        "zh": "气候变化",
        "ja": "気候変動",
        "ko": "기후 변화",
        "ar": "تغير المناخ"
    },
    "artificial intelligence": {
        "es": "inteligencia artificial",
        "fr": "intelligence artificielle",
        "de": "künstliche Intelligenz",
        "it": "intelligenza artificiale",
        "pt": "inteligência artificial",
        "ru": "искусственный интеллект",
        "zh": "人工智能",
        "ja": "人工知能",
        "ko": "인공지능",
        "ar": "الذكاء الاصطناعي"
    },
    "machine learning": {
        "es": "aprendizaje automático",
        "fr": "apprentissage automatique",
        "de": "maschinelles Lernen",
        "it": "apprendimento automatico",
        "pt": "aprendizado de máquina",
        "ru": "машинное обучение",
        "zh": "机器学习",
        "ja": "機械学習",
        "ko": "기계 학습",
        "ar": "تعلم الآلة"
    },
    "renewable energy": {
        "es": "energía renovable",
        "fr": "énergie renouvelable",
        "de": "erneuerbare Energie",
        "it": "energia rinnovabile",
        "pt": "energia renovável",
        "ru": "возобновляемая энергия",
        "zh": "可再生能源",
        "ja": "再生可能エネルギー",
        "ko": "재생 에너지",
        "ar": "الطاقة المتجددة"
    },
    "public health": {
        "es": "salud pública",
        "fr": "santé publique",
        "de": "öffentliche Gesundheit",
        "it": "salute pubblica",
        "pt": "saúde pública",
        "ru": "общественное здравоохранение",
        "zh": "公共卫生",
        "ja": "公衆衛生",
        "ko": "공중 보건",
        "ar": "الصحة العامة"
    },
    "economic development": {
        "es": "desarrollo económico",
        "fr": "développement économique",
        "de": "wirtschaftliche Entwicklung",
        "it": "sviluppo economico",
        "pt": "desenvolvimento econômico",
        "ru": "экономическое развитие",
        "zh": "经济发展",
        "ja": "経済発展",
        "ko": "경제 발��",
        "ar": "التنمية الاقتصادية"
    }
}

# Global storage
_translation_cache: Dict[str, str] = {}
_language_detection_cache: Dict[str, Dict[str, Any]] = {}
_multilingual_results: Dict[str, Dict[str, Any]] = {}


def _detect_language_simple(text: str) -> Dict[str, Any]:
    """Simple language detection based on character patterns."""
    text_sample = text[:1000]  # Use first 1000 characters
    
    detection_scores = {}
    
    for lang_code, pattern in LANGUAGE_PATTERNS.items():
        matches = len(re.findall(pattern, text_sample))
        total_chars = len(re.findall(r'[^\s.,!?;:\'\"()-]', text_sample))
        
        if total_chars > 0:
            score = matches / total_chars
            detection_scores[lang_code] = score
    
    # Find the language with highest score
    if detection_scores:
        detected_lang = max(detection_scores, key=detection_scores.get)
        confidence = detection_scores[detected_lang]
        
        return {
            "detected_language": detected_lang,
            "confidence": confidence,
            "language_name": LANGUAGE_CODES.get(detected_lang, {}).get("name", "Unknown"),
            "all_scores": detection_scores
        }
    
    return {
        "detected_language": "unknown",
        "confidence": 0.0,
        "language_name": "Unknown",
        "all_scores": {}
    }


def _translate_query_terms(query: str, target_language: str) -> str:
    """Translate query terms using predefined translations."""
    translated_query = query.lower()
    
    for english_term, translations in TERM_TRANSLATIONS.items():
        if english_term in translated_query:
            if target_language in translations:
                translated_query = translated_query.replace(english_term, translations[target_language])
    
    return translated_query


def _generate_query_variations(query: str, language: str) -> List[str]:
    """Generate query variations for a specific language."""
    variations = [query]
    
    # Add translated version if available
    translated = _translate_query_terms(query, language)
    if translated != query.lower():
        variations.append(translated)
    
    # Add variations with different word orders (simplified)
    words = query.split()
    if len(words) > 1:
        # Reverse word order
        variations.append(" ".join(reversed(words)))
        
        # Add quoted version for exact phrase
        variations.append(f'"{query}"')
    
    # Add language-specific search operators
    if language in ["de", "nl"]:  # Germanic languages often use compound words
        variations.append(query.replace(" ", ""))
    
    return list(set(variations))  # Remove duplicates


def _create_translation_placeholder(content: str, source_lang: str, target_lang: str) -> str:
    """Create a placeholder translation (in production, use real translation service)."""
    if source_lang == target_lang:
        return content
    
    # Simple placeholder translation
    source_name = LANGUAGE_CODES.get(source_lang, {}).get("name", source_lang)
    target_name = LANGUAGE_CODES.get(target_lang, {}).get("name", target_lang)
    
    # For demonstration, we'll add a translation header
    translation_header = f"[Translated from {source_name} to {target_name}]\n\n"
    
    # In a real implementation, you would use a translation service like:
    # - Google Translate API
    # - Microsoft Translator
    # - DeepL API
    # - Amazon Translate
    
    return translation_header + content


@tool(args_schema=MultilingualSearchInput)
def multilingual_search_query(
    query: str, 
    target_languages: List[str] = None, 
    include_translation: bool = True
) -> str:
    """
    Generate multilingual search queries and coordinate searches across languages.
    
    Args:
        query: Search query to translate and search in multiple languages
        target_languages: Language codes to search in
        include_translation: Include translation of results
    
    Returns:
        JSON string with multilingual search plan
    """
    try:
        if target_languages is None:
            target_languages = ["en", "es", "fr", "de"]
        
        # Validate language codes
        valid_languages = [lang for lang in target_languages if lang in LANGUAGE_CODES]
        invalid_languages = [lang for lang in target_languages if lang not in LANGUAGE_CODES]
        
        # Generate query variations for each language
        multilingual_queries = {}
        
        for lang in valid_languages:
            variations = _generate_query_variations(query, lang)
            multilingual_queries[lang] = {
                "language_name": LANGUAGE_CODES[lang]["name"],
                "native_name": LANGUAGE_CODES[lang]["native"],
                "query_variations": variations,
                "primary_query": variations[0] if variations else query
            }
        
        # Create search strategy
        search_strategy = {
            "original_query": query,
            "target_languages": valid_languages,
            "total_variations": sum(len(q["query_variations"]) for q in multilingual_queries.values()),
            "search_order": _prioritize_language_search_order(valid_languages, query),
            "translation_needed": include_translation
        }
        
        result = {
            "status": "success",
            "multilingual_queries": multilingual_queries,
            "search_strategy": search_strategy,
            "invalid_languages": invalid_languages,
            "recommendations": _generate_multilingual_recommendations(query, valid_languages),
            "timestamp": datetime.now().isoformat(),
            "message": f"Generated multilingual queries for {len(valid_languages)} languages"
        }
        
        # Cache the multilingual plan
        cache_key = hashlib.md5(f"{query}{''.join(valid_languages)}".encode()).hexdigest()
        _multilingual_results[cache_key] = result
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Multilingual search generation failed: {str(e)}"
        })


def _prioritize_language_search_order(languages: List[str], query: str) -> List[str]:
    """Prioritize language search order based on query content and language importance."""
    # Detect query language
    detection = _detect_language_simple(query)
    detected_lang = detection["detected_language"]
    
    # Priority order: detected language first, then by global usage
    language_priority = {
        "en": 10,  # English - most content
        "zh": 9,   # Chinese - large population
        "es": 8,   # Spanish - widely spoken
        "fr": 7,   # French - international
        "de": 6,   # German - scientific content
        "ja": 5,   # Japanese - technology
        "pt": 4,   # Portuguese - growing
        "ru": 4,   # Russian - scientific
        "ar": 3,   # Arabic - regional
        "it": 3,   # Italian - regional
        "ko": 2,   # Korean - technology
        "hi": 2    # Hindi - large population
    }
    
    # Sort languages by priority, with detected language first
    sorted_languages = sorted(languages, key=lambda x: (
        -1 if x == detected_lang else language_priority.get(x, 1)
    ), reverse=True)
    
    return sorted_languages


def _generate_multilingual_recommendations(query: str, languages: List[str]) -> List[str]:
    """Generate recommendations for multilingual search."""
    recommendations = []
    
    # Detect query language
    detection = _detect_language_simple(query)
    detected_lang = detection["detected_language"]
    
    if detected_lang != "en" and "en" not in languages:
        recommendations.append("Consider adding English (en) for broader coverage")
    
    if len(languages) < 3:
        recommendations.append("Consider searching in more languages for comprehensive results")
    
    # Suggest relevant languages based on query content
    query_lower = query.lower()
    
    if any(term in query_lower for term in ["europe", "european", "eu"]):
        eu_languages = ["en", "fr", "de", "es", "it"]
        missing_eu = [lang for lang in eu_languages if lang not in languages]
        if missing_eu:
            recommendations.append(f"Consider European languages: {', '.join(missing_eu)}")
    
    if any(term in query_lower for term in ["asia", "asian"]):
        asian_languages = ["zh", "ja", "ko", "hi"]
        missing_asian = [lang for lang in asian_languages if lang not in languages]
        if missing_asian:
            recommendations.append(f"Consider Asian languages: {', '.join(missing_asian)}")
    
    if any(term in query_lower for term in ["science", "research", "study"]):
        science_languages = ["en", "de", "fr", "ja"]
        missing_science = [lang for lang in science_languages if lang not in languages]
        if missing_science:
            recommendations.append(f"Consider scientific languages: {', '.join(missing_science)}")
    
    return recommendations


@tool(args_schema=TranslateContentInput)
def translate_research_content(
    content: str, 
    source_language: str, 
    target_language: str = "en", 
    preserve_formatting: bool = True
) -> str:
    """
    Translate research content from source language to target language.
    
    Args:
        content: Content to translate
        source_language: Source language code
        target_language: Target language code
        preserve_formatting: Preserve original formatting
    
    Returns:
        JSON string with translation result
    """
    try:
        # Validate language codes
        if source_language not in LANGUAGE_CODES:
            return json.dumps({
                "status": "failure",
                "message": f"Unsupported source language: {source_language}"
            })
        
        if target_language not in LANGUAGE_CODES:
            return json.dumps({
                "status": "failure",
                "message": f"Unsupported target language: {target_language}"
            })
        
        # Check cache
        cache_key = hashlib.md5(f"{content[:100]}{source_language}{target_language}".encode()).hexdigest()
        
        if cache_key in _translation_cache:
            cached_translation = _translation_cache[cache_key]
            return json.dumps({
                "status": "cache_hit",
                "original_content": content[:200] + "..." if len(content) > 200 else content,
                "translated_content": cached_translation,
                "source_language": source_language,
                "target_language": target_language,
                "message": "Retrieved cached translation"
            })
        
        # Perform translation (placeholder implementation)
        translated_content = _create_translation_placeholder(content, source_language, target_language)
        
        # Preserve formatting if requested
        if preserve_formatting:
            # Simple formatting preservation
            if content.startswith("# "):
                translated_content = "# " + translated_content
            elif content.startswith("## "):
                translated_content = "## " + translated_content
        
        # Cache translation
        _translation_cache[cache_key] = translated_content
        
        # Analyze translation quality (placeholder)
        translation_quality = _assess_translation_quality(content, translated_content, source_language, target_language)
        
        result = {
            "status": "success",
            "original_content": content,
            "translated_content": translated_content,
            "source_language": source_language,
            "target_language": target_language,
            "source_language_name": LANGUAGE_CODES[source_language]["name"],
            "target_language_name": LANGUAGE_CODES[target_language]["name"],
            "translation_quality": translation_quality,
            "preserve_formatting": preserve_formatting,
            "timestamp": datetime.now().isoformat(),
            "message": f"Translated content from {LANGUAGE_CODES[source_language]['name']} to {LANGUAGE_CODES[target_language]['name']}"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Translation failed: {str(e)}"
        })


def _assess_translation_quality(original: str, translated: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
    """Assess translation quality (placeholder implementation)."""
    return {
        "confidence_score": 85,  # Placeholder
        "length_ratio": len(translated) / len(original) if original else 1,
        "estimated_accuracy": "high",  # Placeholder
        "notes": ["This is a placeholder translation", "Use professional translation service for production"]
    }


@tool(args_schema=DetectLanguageInput)
def detect_content_language(text: str, confidence_threshold: float = 0.8) -> str:
    """
    Detect the language of text content.
    
    Args:
        text: Text to detect language for
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        JSON string with language detection result
    """
    try:
        # Check cache
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        
        if cache_key in _language_detection_cache:
            cached_result = _language_detection_cache[cache_key]
            return json.dumps({
                "status": "cache_hit",
                **cached_result,
                "message": "Retrieved cached language detection"
            })
        
        # Perform language detection
        detection_result = _detect_language_simple(text)
        
        # Determine if detection meets confidence threshold
        meets_threshold = detection_result["confidence"] >= confidence_threshold
        
        # Additional analysis
        text_analysis = {
            "text_length": len(text),
            "character_count": len(re.findall(r'[^\s]', text)),
            "word_count": len(text.split()),
            "has_special_characters": bool(re.search(r'[^\x00-\x7F]', text)),
            "sample_text": text[:100] + "..." if len(text) > 100 else text
        }
        
        result = {
            "status": "success",
            "detection_result": detection_result,
            "meets_threshold": meets_threshold,
            "confidence_threshold": confidence_threshold,
            "text_analysis": text_analysis,
            "recommendations": _generate_language_recommendations(detection_result, meets_threshold),
            "timestamp": datetime.now().isoformat(),
            "message": f"Detected language: {detection_result['language_name']} (confidence: {detection_result['confidence']:.2f})"
        }
        
        # Cache result
        _language_detection_cache[cache_key] = result
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Language detection failed: {str(e)}"
        })


def _generate_language_recommendations(detection_result: Dict[str, Any], meets_threshold: bool) -> List[str]:
    """Generate recommendations based on language detection."""
    recommendations = []
    
    if not meets_threshold:
        recommendations.append("Low confidence detection - consider manual verification")
        recommendations.append("Try with longer text sample for better accuracy")
    
    detected_lang = detection_result["detected_language"]
    
    if detected_lang == "unknown":
        recommendations.append("Could not detect language - text may be mixed language or too short")
        recommendations.append("Consider using language-specific search terms")
    elif detected_lang != "en":
        recommendations.append("Consider translating to English for broader search coverage")
        recommendations.append(f"Search in native language ({detected_lang}) for local sources")
    
    return recommendations


@tool(args_schema=GenerateQueryVariationsInput)
def generate_multilingual_variations(
    query: str, 
    languages: List[str], 
    include_synonyms: bool = True
) -> str:
    """
    Generate query variations in multiple languages.
    
    Args:
        query: Original query to generate variations for
        languages: Languages to generate variations in
        include_synonyms: Include synonym variations
    
    Returns:
        JSON string with query variations
    """
    try:
        # Validate languages
        valid_languages = [lang for lang in languages if lang in LANGUAGE_CODES]
        invalid_languages = [lang for lang in languages if lang not in LANGUAGE_CODES]
        
        variations_by_language = {}
        
        for lang in valid_languages:
            # Generate basic variations
            variations = _generate_query_variations(query, lang)
            
            # Add synonym variations if requested
            if include_synonyms:
                synonym_variations = _generate_synonym_variations(query, lang)
                variations.extend(synonym_variations)
            
            # Remove duplicates and empty strings
            variations = list(set([v for v in variations if v.strip()]))
            
            variations_by_language[lang] = {
                "language_name": LANGUAGE_CODES[lang]["name"],
                "native_name": LANGUAGE_CODES[lang]["native"],
                "variations": variations,
                "variation_count": len(variations)
            }
        
        # Generate search strategy
        total_variations = sum(data["variation_count"] for data in variations_by_language.values())
        
        search_strategy = {
            "recommended_order": _prioritize_language_search_order(valid_languages, query),
            "parallel_search": len(valid_languages) <= 4,  # Parallel if 4 or fewer languages
            "estimated_search_time": total_variations * 2,  # Rough estimate in seconds
            "optimization_tips": _generate_search_optimization_tips(total_variations, valid_languages)
        }
        
        result = {
            "status": "success",
            "original_query": query,
            "variations_by_language": variations_by_language,
            "search_strategy": search_strategy,
            "invalid_languages": invalid_languages,
            "total_variations": total_variations,
            "include_synonyms": include_synonyms,
            "timestamp": datetime.now().isoformat(),
            "message": f"Generated {total_variations} query variations across {len(valid_languages)} languages"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Query variation generation failed: {str(e)}"
        })


def _generate_synonym_variations(query: str, language: str) -> List[str]:
    """Generate synonym variations for a query in a specific language."""
    # This is a simplified implementation
    # In production, you'd use a thesaurus or synonym API
    
    variations = []
    query_lower = query.lower()
    
    # English synonyms
    if language == "en":
        synonym_map = {
            "research": ["study", "investigation", "analysis"],
            "study": ["research", "investigation", "examination"],
            "analysis": ["examination", "evaluation", "assessment"],
            "development": ["growth", "progress", "advancement"],
            "technology": ["tech", "innovation", "digital"],
            "health": ["medical", "healthcare", "wellness"],
            "environment": ["environmental", "ecological", "green"]
        }
        
        for word, synonyms in synonym_map.items():
            if word in query_lower:
                for synonym in synonyms:
                    variations.append(query_lower.replace(word, synonym))
    
    # Add language-specific variations
    elif language == "es":
        # Spanish variations
        if "investigación" in query_lower:
            variations.append(query_lower.replace("investigación", "estudio"))
    elif language == "fr":
        # French variations
        if "recherche" in query_lower:
            variations.append(query_lower.replace("recherche", "étude"))
    
    return variations


def _generate_search_optimization_tips(total_variations: int, languages: List[str]) -> List[str]:
    """Generate tips for optimizing multilingual search."""
    tips = []
    
    if total_variations > 20:
        tips.append("Consider reducing query variations to improve search speed")
    
    if len(languages) > 5:
        tips.append("Focus on 3-5 most relevant languages for better results")
    
    if "en" not in languages:
        tips.append("Include English for maximum content coverage")
    
    tips.append("Start with highest-priority languages first")
    tips.append("Use parallel search for multiple languages when possible")
    
    return tips


@tool
def get_multilingual_statistics() -> str:
    """
    Get statistics about multilingual research activities.
    
    Returns:
        JSON string with multilingual statistics
    """
    try:
        total_translations = len(_translation_cache)
        total_detections = len(_language_detection_cache)
        total_multilingual_searches = len(_multilingual_results)
        
        # Analyze language usage
        language_usage = {}
        for result in _multilingual_results.values():
            for lang in result.get("search_strategy", {}).get("target_languages", []):
                language_usage[lang] = language_usage.get(lang, 0) + 1
        
        # Most popular languages
        popular_languages = sorted(language_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        
        result = {
            "status": "success",
            "statistics": {
                "total_translations": total_translations,
                "total_language_detections": total_detections,
                "total_multilingual_searches": total_multilingual_searches,
                "supported_languages": len(LANGUAGE_CODES),
                "popular_languages": [
                    {
                        "code": code,
                        "name": LANGUAGE_CODES.get(code, {}).get("name", "Unknown"),
                        "usage_count": count
                    }
                    for code, count in popular_languages
                ],
                "available_term_translations": len(TERM_TRANSLATIONS)
            },
            "message": f"Multilingual statistics: {total_translations} translations, {total_detections} detections"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get multilingual statistics: {str(e)}"
        })


@tool
def clear_multilingual_cache() -> str:
    """
    Clear multilingual research cache.
    
    Returns:
        JSON string with cleanup results
    """
    try:
        global _translation_cache, _language_detection_cache, _multilingual_results
        
        translation_count = len(_translation_cache)
        detection_count = len(_language_detection_cache)
        search_count = len(_multilingual_results)
        
        _translation_cache.clear()
        _language_detection_cache.clear()
        _multilingual_results.clear()
        
        total_cleared = translation_count + detection_count + search_count
        
        return json.dumps({
            "status": "success",
            "cleared_items": {
                "translations": translation_count,
                "language_detections": detection_count,
                "multilingual_searches": search_count,
                "total": total_cleared
            },
            "message": f"Cleared {total_cleared} multilingual cache items"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Cache cleanup failed: {str(e)}"
        })