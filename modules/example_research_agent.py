"""
Example Research Agent Module
Demonstrates how to create a modular agent using the new module system.

@version: 1.0.0
@author: AgentKen Team
@id: research_agent_v2
"""

import time
import logging
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from module_system import AgentModule, ModuleCapability, ModuleMetadata, ModuleType, ModuleDependency, DependencyType


class ResearchAgentModule(AgentModule):
    """Modular research agent with web search and analysis capabilities"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        super().__init__(module_id, config)
        self.search_engine = config.get("search_engine", "duckduckgo") if config else "duckduckgo"
        self.max_results = config.get("max_results", 10) if config else 10
        self.timeout = config.get("timeout", 30) if config else 30
        
        # Initialize metadata
        self.metadata = ModuleMetadata(
            id="research_agent_v2",
            name="Research Agent v2",
            version="1.0.0",
            module_type=ModuleType.AGENT,
            description="Advanced research agent with web search and content analysis",
            author="AgentKen Team",
            license="MIT",
            homepage="https://agentken.io/modules/research-agent",
            documentation="https://docs.agentken.io/modules/research-agent",
            
            dependencies=[
                ModuleDependency(
                    name="web_search_tool",
                    version="1.0.0",
                    dependency_type=DependencyType.OPTIONAL,
                    description="Enhanced web search capabilities"
                )
            ],
            
            python_requirements=[
                "requests>=2.25.0",
                "beautifulsoup4>=4.9.0",
                "lxml>=4.6.0"
            ],
            
            capabilities=[
                ModuleCapability(
                    name="web_search",
                    description="Search the web for information on a given topic",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "language": {"type": "string", "default": "en"}
                        },
                        "required": ["query"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                        "relevance_score": {"type": "number"}
                                    }
                                }
                            },
                            "total_results": {"type": "integer"},
                            "search_time": {"type": "number"}
                        }
                    },
                    tags=["search", "web", "information"]
                ),
                
                ModuleCapability(
                    name="analyze_content",
                    description="Analyze web content for key information and insights",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to analyze"},
                            "content": {"type": "string", "description": "Content to analyze"},
                            "analysis_type": {"type": "string", "enum": ["summary", "keywords", "sentiment", "full"]}
                        }
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "sentiment": {"type": "string"},
                            "key_points": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"}
                        }
                    },
                    tags=["analysis", "content", "nlp"]
                ),
                
                ModuleCapability(
                    name="research_topic",
                    description="Comprehensive research on a topic with multiple sources",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Research topic"},
                            "depth": {"type": "string", "enum": ["basic", "detailed", "comprehensive"]},
                            "sources": {"type": "integer", "default": 5}
                        },
                        "required": ["topic"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "summary": {"type": "string"},
                            "key_findings": {"type": "array", "items": {"type": "string"}},
                            "sources": {"type": "array"},
                            "confidence": {"type": "number"},
                            "research_time": {"type": "number"}
                        }
                    },
                    tags=["research", "comprehensive", "analysis"]
                )
            ],
            
            configuration_schema={
                "type": "object",
                "properties": {
                    "search_engine": {"type": "string", "enum": ["duckduckgo", "google", "bing"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
                    "timeout": {"type": "integer", "minimum": 5, "maximum": 120},
                    "api_key": {"type": "string", "description": "API key for search services"}
                }
            },
            
            tags=["research", "web", "analysis", "agent"],
            category="research",
            priority=1
        )
    
    def initialize(self) -> bool:
        """Initialize the research agent module"""
        try:
            # Validate configuration
            if not self.validate_config(self.config):
                self.logger.error("Invalid configuration")
                return False
            
            # Test search engine connectivity
            if not self._test_connectivity():
                self.logger.warning("Search engine connectivity test failed")
                # Continue anyway as it might be temporary
            
            self.logger.info("Research agent module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize research agent: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the research agent module"""
        try:
            # Clean up any resources
            self.logger.info("Research agent module shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown research agent: {e}")
            return False
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """Return list of capabilities"""
        return self.metadata.capabilities
    
    def execute(self, capability: str, **kwargs) -> Any:
        """Execute a specific capability"""
        if capability == "web_search":
            return self._web_search(**kwargs)
        elif capability == "analyze_content":
            return self._analyze_content(**kwargs)
        elif capability == "research_topic":
            return self._research_topic(**kwargs)
        else:
            raise ValueError(f"Unknown capability: {capability}")
    
    def create_agent(self) -> Any:
        """Create and return the agent instance"""
        return ResearchAgent(self.config)
    
    def _web_search(self, query: str, max_results: int = None, language: str = "en") -> Dict[str, Any]:
        """Perform web search"""
        start_time = time.time()
        max_results = max_results or self.max_results
        
        try:
            if self.search_engine == "duckduckgo":
                results = self._duckduckgo_search(query, max_results, language)
            else:
                # Fallback to DuckDuckGo
                results = self._duckduckgo_search(query, max_results, language)
            
            search_time = time.time() - start_time
            
            return {
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "query": query,
                "search_engine": self.search_engine
            }
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            raise
    
    def _analyze_content(self, url: str = None, content: str = None, 
                        analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze web content"""
        try:
            # Get content if URL provided
            if url and not content:
                content = self._fetch_content(url)
            
            if not content:
                raise ValueError("No content to analyze")
            
            # Perform analysis based on type
            analysis = {}
            
            if analysis_type in ["summary", "full"]:
                analysis["summary"] = self._generate_summary(content)
            
            if analysis_type in ["keywords", "full"]:
                analysis["keywords"] = self._extract_keywords(content)
            
            if analysis_type in ["sentiment", "full"]:
                analysis["sentiment"] = self._analyze_sentiment(content)
            
            if analysis_type == "full":
                analysis["key_points"] = self._extract_key_points(content)
                analysis["confidence"] = self._calculate_confidence(content)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            raise
    
    def _research_topic(self, topic: str, depth: str = "basic", sources: int = 5) -> Dict[str, Any]:
        """Comprehensive topic research"""
        start_time = time.time()
        
        try:
            # Perform initial search
            search_results = self._web_search(topic, max_results=sources * 2)
            
            # Analyze top sources
            analyzed_sources = []
            key_findings = []
            
            for result in search_results["results"][:sources]:
                try:
                    analysis = self._analyze_content(url=result["url"], analysis_type="full")
                    analyzed_sources.append({
                        "url": result["url"],
                        "title": result["title"],
                        "analysis": analysis
                    })
                    
                    if "key_points" in analysis:
                        key_findings.extend(analysis["key_points"])
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze source {result['url']}: {e}")
                    continue
            
            # Generate comprehensive summary
            summary = self._generate_research_summary(topic, analyzed_sources)
            
            # Calculate overall confidence
            confidences = [s["analysis"].get("confidence", 0.5) for s in analyzed_sources 
                          if "confidence" in s["analysis"]]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            research_time = time.time() - start_time
            
            return {
                "topic": topic,
                "summary": summary,
                "key_findings": list(set(key_findings))[:10],  # Top 10 unique findings
                "sources": analyzed_sources,
                "confidence": overall_confidence,
                "research_time": research_time,
                "depth": depth
            }
            
        except Exception as e:
            self.logger.error(f"Topic research failed: {e}")
            raise
    
    def _duckduckgo_search(self, query: str, max_results: int, language: str) -> List[Dict[str, Any]]:
        """Perform DuckDuckGo search"""
        # Simplified DuckDuckGo search implementation
        # In a real implementation, you would use the DuckDuckGo API or scraping
        
        results = []
        
        # Mock results for demonstration
        for i in range(min(max_results, 5)):
            results.append({
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a sample snippet for result {i+1} about {query}. It contains relevant information about the search topic.",
                "relevance_score": 0.9 - (i * 0.1)
            })
        
        return results
    
    def _fetch_content(self, url: str) -> str:
        """Fetch content from URL"""
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to fetch content from {url}: {e}")
            return ""
    
    def _generate_summary(self, content: str) -> str:
        """Generate content summary"""
        # Simplified summary generation
        sentences = content.split('.')[:3]  # First 3 sentences
        return '. '.join(sentences).strip() + '.'
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simplified keyword extraction
        words = content.lower().split()
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        
        return [word for word, count in word_counts.most_common(10)]
    
    def _analyze_sentiment(self, content: str) -> str:
        """Analyze content sentiment"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'worst', 'problem', 'issue']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        # Simplified key point extraction
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Return first few sentences as key points
        return sentences[:5]
    
    def _calculate_confidence(self, content: str) -> float:
        """Calculate analysis confidence"""
        # Simplified confidence calculation based on content length and structure
        if len(content) < 100:
            return 0.3
        elif len(content) < 500:
            return 0.6
        elif len(content) < 1000:
            return 0.8
        else:
            return 0.9
    
    def _generate_research_summary(self, topic: str, sources: List[Dict[str, Any]]) -> str:
        """Generate comprehensive research summary"""
        if not sources:
            return f"No reliable sources found for research on '{topic}'."
        
        summary_parts = [f"Research summary for '{topic}':"]
        
        for i, source in enumerate(sources[:3], 1):
            if "summary" in source["analysis"]:
                summary_parts.append(f"{i}. {source['analysis']['summary']}")
        
        return " ".join(summary_parts)
    
    def _test_connectivity(self) -> bool:
        """Test search engine connectivity"""
        try:
            # Simple connectivity test
            response = requests.get("https://duckduckgo.com", timeout=5)
            return response.status_code == 200
        except:
            return False


class ResearchAgent:
    """Traditional agent class that can be used independently"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("research_agent")
    
    def search(self, query: str) -> Dict[str, Any]:
        """Search for information"""
        # Implementation would use the module's capabilities
        return {"query": query, "results": []}
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze content"""
        # Implementation would use the module's capabilities
        return {"analysis": "content analysis"}
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Research a topic"""
        # Implementation would use the module's capabilities
        return {"topic": topic, "findings": []}


# Module metadata for discovery
MODULE_METADATA = {
    "id": "research_agent_v2",
    "name": "Research Agent v2",
    "version": "1.0.0",
    "module_type": "agent",
    "description": "Advanced research agent with web search and content analysis",
    "author": "AgentKen Team",
    "license": "MIT",
    "capabilities": [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "tags": ["search", "web"]
        },
        {
            "name": "analyze_content", 
            "description": "Analyze web content",
            "tags": ["analysis", "content"]
        },
        {
            "name": "research_topic",
            "description": "Comprehensive topic research",
            "tags": ["research", "comprehensive"]
        }
    ],
    "dependencies": [],
    "python_requirements": ["requests>=2.25.0", "beautifulsoup4>=4.9.0"],
    "tags": ["research", "web", "analysis"]
}


def get_metadata():
    """Function to get module metadata"""
    return MODULE_METADATA


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Create module instance
    module = ResearchAgentModule("research_agent_v2")
    
    # Initialize
    if module.initialize():
        print("✅ Module initialized successfully")
        
        # Test capabilities
        try:
            # Test web search
            search_result = module.execute("web_search", query="artificial intelligence")
            print(f"Search results: {len(search_result['results'])} found")
            
            # Test content analysis
            analysis_result = module.execute("analyze_content", 
                                           content="This is a test content for analysis. It contains information about AI and machine learning.",
                                           analysis_type="full")
            print(f"Analysis completed: {analysis_result.get('sentiment', 'unknown')} sentiment")
            
            # Test topic research
            research_result = module.execute("research_topic", topic="machine learning", depth="basic")
            print(f"Research completed: {len(research_result['key_findings'])} key findings")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        # Shutdown
        module.shutdown()
    else:
        print("❌ Module initialization failed")