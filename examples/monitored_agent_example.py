
from core.performance_decorators import monitor_agent_execution, MonitoredAgent
import time

class ExampleAgent(MonitoredAgent):
    """Example agent with performance monitoring"""
    
    def __init__(self):
        super().__init__("example_agent")
    
    def research_topic(self, topic: str) -> dict:
        """Research a topic with monitoring"""
        return self.execute_with_monitoring("research_topic", self._do_research, topic)
    
    def _do_research(self, topic: str) -> dict:
        # Simulate research work
        time.sleep(0.2)
        return {"topic": topic, "results": ["result1", "result2"]}

# Alternative using decorators
@monitor_agent_execution("research_agent", "web_search")
def search_web(query: str) -> dict:
    time.sleep(0.1)
    return {"query": query, "results": ["web result 1", "web result 2"]}
