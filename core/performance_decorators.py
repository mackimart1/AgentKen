"""
Performance Monitoring Decorators and Middleware
Easy integration of performance monitoring into existing agents and tools.
"""

import functools
import time
import logging
import threading
from typing import Any, Callable, Dict, Optional
from performance_monitor import PerformanceMonitor, ComponentType, MetricType


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def initialize_performance_monitoring(db_path: str = "performance_metrics.db") -> PerformanceMonitor:
    """Initialize the global performance monitor"""
    global _performance_monitor
    
    with _monitor_lock:
        if _performance_monitor is None:
            _performance_monitor = PerformanceMonitor(db_path)
            _performance_monitor.start()
            logging.info("Performance monitoring initialized")
        
        return _performance_monitor


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance"""
    return _performance_monitor


def monitor_agent_execution(agent_id: str, operation: str = None):
    """Decorator to monitor agent execution performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if not monitor:
                # If monitoring is not initialized, just execute the function
                return func(*args, **kwargs)
            
            op_name = operation or func.__name__
            
            with monitor.track_execution(agent_id, ComponentType.AGENT, op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_tool_execution(tool_id: str, operation: str = None):
    """Decorator to monitor tool execution performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if not monitor:
                return func(*args, **kwargs)
            
            op_name = operation or func.__name__
            
            with monitor.track_execution(tool_id, ComponentType.TOOL, op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_workflow_execution(workflow_id: str, operation: str = None):
    """Decorator to monitor workflow execution performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if not monitor:
                return func(*args, **kwargs)
            
            op_name = operation or func.__name__
            
            with monitor.track_execution(workflow_id, ComponentType.WORKFLOW, op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PerformanceMiddleware:
    """Middleware class for monitoring performance"""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.monitor = get_performance_monitor()
    
    def track_execution(self, operation: str):
        """Context manager for tracking execution"""
        if self.monitor:
            return self.monitor.track_execution(self.component_id, self.component_type, operation)
        else:
            return DummyTracker()
    
    def record_metric(self, metric_type: MetricType, value: float, unit: str,
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a custom metric"""
        if self.monitor:
            self.monitor.collector.record_metric(
                self.component_id, self.component_type, metric_type, value, unit, tags, metadata
            )


class DummyTracker:
    """Dummy tracker for when monitoring is not available"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MonitoredAgent:
    """Base class for agents with built-in performance monitoring"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.middleware = PerformanceMiddleware(agent_id, ComponentType.AGENT)
    
    def execute_with_monitoring(self, operation: str, func: Callable, *args, **kwargs):
        """Execute a function with performance monitoring"""
        with self.middleware.track_execution(operation):
            return func(*args, **kwargs)
    
    def record_custom_metric(self, metric_type: MetricType, value: float, unit: str,
                           tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a custom metric for this agent"""
        self.middleware.record_metric(metric_type, value, unit, tags, metadata)


class MonitoredTool:
    """Base class for tools with built-in performance monitoring"""
    
    def __init__(self, tool_id: str):
        self.tool_id = tool_id
        self.middleware = PerformanceMiddleware(tool_id, ComponentType.TOOL)
    
    def execute_with_monitoring(self, operation: str, func: Callable, *args, **kwargs):
        """Execute a function with performance monitoring"""
        with self.middleware.track_execution(operation):
            return func(*args, **kwargs)
    
    def record_custom_metric(self, metric_type: MetricType, value: float, unit: str,
                           tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a custom metric for this tool"""
        self.middleware.record_metric(metric_type, value, unit, tags, metadata)


def add_performance_monitoring_to_class(cls, component_id: str, component_type: ComponentType):
    """Add performance monitoring to an existing class"""
    
    # Store original methods
    original_methods = {}
    
    # Find methods to monitor (exclude private methods and special methods)
    methods_to_monitor = [
        name for name, method in cls.__dict__.items()
        if callable(method) and not name.startswith('_')
    ]
    
    for method_name in methods_to_monitor:
        original_method = getattr(cls, method_name)
        original_methods[method_name] = original_method
        
        # Create monitored version
        if component_type == ComponentType.AGENT:
            monitored_method = monitor_agent_execution(component_id, method_name)(original_method)
        elif component_type == ComponentType.TOOL:
            monitored_method = monitor_tool_execution(component_id, method_name)(original_method)
        else:
            monitored_method = monitor_workflow_execution(component_id, method_name)(original_method)
        
        # Replace the method
        setattr(cls, method_name, monitored_method)
    
    # Add a method to restore original methods if needed
    def restore_original_methods():
        for method_name, original_method in original_methods.items():
            setattr(cls, method_name, original_method)
    
    cls._restore_original_methods = restore_original_methods
    
    return cls


# Convenience functions for common monitoring patterns
def track_api_call(component_id: str, api_name: str):
    """Track an API call"""
    monitor = get_performance_monitor()
    if monitor:
        return monitor.track_execution(component_id, ComponentType.SYSTEM, f"api_call_{api_name}")
    return DummyTracker()


def track_database_operation(component_id: str, operation: str):
    """Track a database operation"""
    monitor = get_performance_monitor()
    if monitor:
        return monitor.track_execution(component_id, ComponentType.SYSTEM, f"db_{operation}")
    return DummyTracker()


def track_file_operation(component_id: str, operation: str):
    """Track a file operation"""
    monitor = get_performance_monitor()
    if monitor:
        return monitor.track_execution(component_id, ComponentType.SYSTEM, f"file_{operation}")
    return DummyTracker()


def record_memory_usage(component_id: str, memory_mb: float):
    """Record memory usage metric"""
    monitor = get_performance_monitor()
    if monitor:
        monitor.collector.record_metric(
            component_id, ComponentType.SYSTEM, MetricType.MEMORY_USAGE, memory_mb, "MB"
        )


def record_cpu_usage(component_id: str, cpu_percent: float):
    """Record CPU usage metric"""
    monitor = get_performance_monitor()
    if monitor:
        monitor.collector.record_metric(
            component_id, ComponentType.SYSTEM, MetricType.CPU_USAGE, cpu_percent, "percent"
        )


def record_throughput(component_id: str, requests_per_second: float):
    """Record throughput metric"""
    monitor = get_performance_monitor()
    if monitor:
        monitor.collector.record_metric(
            component_id, ComponentType.SYSTEM, MetricType.THROUGHPUT, requests_per_second, "rps"
        )


def record_error_count(component_id: str, error_count: int):
    """Record error count metric"""
    monitor = get_performance_monitor()
    if monitor:
        monitor.collector.record_metric(
            component_id, ComponentType.SYSTEM, MetricType.ERROR_COUNT, error_count, "count"
        )


# Example usage and integration helpers
class PerformanceAwareAgent(MonitoredAgent):
    """Example of an agent with built-in performance monitoring"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task with automatic performance monitoring"""
        return self.execute_with_monitoring("process_task", self._do_process_task, task_data)
    
    def _do_process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actual task processing logic"""
        # Simulate some work
        time.sleep(0.1)
        
        # Record custom metrics
        self.record_custom_metric(MetricType.THROUGHPUT, 10.0, "tasks_per_second")
        
        return {"status": "completed", "result": "task processed"}


class PerformanceAwareTool(MonitoredTool):
    """Example of a tool with built-in performance monitoring"""
    
    def __init__(self, tool_id: str):
        super().__init__(tool_id)
    
    def execute_operation(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an operation with automatic performance monitoring"""
        return self.execute_with_monitoring("execute_operation", self._do_execute, operation_data)
    
    def _do_execute(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actual operation execution logic"""
        # Simulate some work
        time.sleep(0.05)
        
        # Record custom metrics
        self.record_custom_metric(MetricType.RESOURCE_USAGE, 25.0, "percent")
        
        return {"status": "success", "output": "operation completed"}


# Integration with existing AgentKen components
def integrate_with_existing_agent(agent_class, agent_id: str):
    """Integrate performance monitoring with an existing agent class"""
    return add_performance_monitoring_to_class(agent_class, agent_id, ComponentType.AGENT)


def integrate_with_existing_tool(tool_class, tool_id: str):
    """Integrate performance monitoring with an existing tool class"""
    return add_performance_monitoring_to_class(tool_class, tool_id, ComponentType.TOOL)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize performance monitoring
    monitor = initialize_performance_monitoring()
    
    # Example 1: Using decorators
    @monitor_agent_execution("example_agent", "search")
    def search_web(query: str) -> Dict[str, Any]:
        time.sleep(0.2)  # Simulate work
        return {"results": [f"Result for {query}"]}
    
    @monitor_tool_execution("web_scraper", "scrape")
    def scrape_page(url: str) -> str:
        time.sleep(0.1)  # Simulate work
        return f"Content from {url}"
    
    # Example 2: Using classes
    agent = PerformanceAwareAgent("test_agent")
    tool = PerformanceAwareTool("test_tool")
    
    # Execute some operations
    print("Executing monitored operations...")
    
    search_result = search_web("AI frameworks")
    scrape_result = scrape_page("https://example.com")
    
    agent_result = agent.process_task({"task": "analyze data"})
    tool_result = tool.execute_operation({"operation": "transform data"})
    
    # Wait for metrics to be processed
    time.sleep(1)
    
    # Generate performance report
    overview = monitor.dashboard.generate_system_overview()
    print(f"\nPerformance Overview:")
    print(f"Total Executions: {overview['system_metrics']['total_executions']}")
    print(f"Success Rate: {overview['system_metrics']['system_success_rate']:.1f}%")
    print(f"Average Latency: {overview['system_metrics']['avg_system_latency']:.1f}ms")
    
    # Stop monitoring
    monitor.stop()