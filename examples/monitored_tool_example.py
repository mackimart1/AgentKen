
from core.performance_decorators import monitor_tool_execution, MonitoredTool
import time

class ExampleTool(MonitoredTool):
    """Example tool with performance monitoring"""
    
    def __init__(self):
        super().__init__("example_tool")
    
    def process_data(self, data: dict) -> dict:
        """Process data with monitoring"""
        return self.execute_with_monitoring("process_data", self._do_process, data)
    
    def _do_process(self, data: dict) -> dict:
        # Simulate data processing
        time.sleep(0.15)
        return {"processed": True, "input_size": len(str(data))}

# Alternative using decorators
@monitor_tool_execution("data_processor", "transform")
def transform_data(data: list) -> list:
    time.sleep(0.05)
    return [item.upper() for item in data]
