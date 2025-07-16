# AgentKen Performance Monitoring System

## Overview

The AgentKen Performance Monitoring System provides comprehensive performance tracking, alerting, and visualization for agents, tools, and workflows. It captures key metrics such as latency, success/failure rates, throughput, and resource usage to help identify and address system bottlenecks.

## Features

### 🔍 **Metrics Collection**
- **Latency Tracking**: Execution time for all operations
- **Success/Failure Rates**: Track operation outcomes
- **Throughput Monitoring**: Requests per second and processing rates
- **Resource Usage**: Memory and CPU utilization
- **Custom Metrics**: Application-specific measurements

### 🚨 **Intelligent Alerting**
- **Configurable Thresholds**: Set custom alert rules
- **Multiple Severity Levels**: Info, Warning, Error, Critical
- **Real-time Notifications**: Console, log, email, webhook support
- **Alert Management**: Acknowledge and resolve alerts

### 📊 **Performance Dashboard**
- **Real-time Visualization**: Live performance metrics
- **Component Analysis**: Detailed per-component statistics
- **Trend Analysis**: Historical performance trends
- **Bottleneck Identification**: Automatic problem detection
- **Health Scoring**: Overall system health assessment

### 🔧 **Easy Integration**
- **Decorators**: Simple function decoration for monitoring
- **Base Classes**: Inherit from monitored agent/tool classes
- **Middleware**: Flexible integration options
- **Auto-discovery**: Automatic component detection

## Quick Start

### 1. Setup and Installation

```bash
# Run the setup script
python setup_performance_monitoring.py

# Start with dashboard
python setup_performance_monitoring.py --dashboard
```

### 2. Basic Integration

#### Using Decorators

```python
from core.performance_decorators import monitor_agent_execution, monitor_tool_execution

# Monitor an agent method
@monitor_agent_execution("research_agent", "web_search")
def search_web(query: str) -> dict:
    # Your agent logic here
    return {"results": ["result1", "result2"]}

# Monitor a tool method
@monitor_tool_execution("data_processor", "transform")
def transform_data(data: list) -> list:
    # Your tool logic here
    return [item.upper() for item in data]
```

#### Using Base Classes

```python
from core.performance_decorators import MonitoredAgent, MonitoredTool

class MyAgent(MonitoredAgent):
    def __init__(self):
        super().__init__("my_agent")
    
    def process_task(self, task_data: dict) -> dict:
        return self.execute_with_monitoring("process_task", self._do_process, task_data)
    
    def _do_process(self, task_data: dict) -> dict:
        # Your processing logic
        return {"status": "completed"}

class MyTool(MonitoredTool):
    def __init__(self):
        super().__init__("my_tool")
    
    def execute_operation(self, data: dict) -> dict:
        return self.execute_with_monitoring("execute", self._do_execute, data)
    
    def _do_execute(self, data: dict) -> dict:
        # Your tool logic
        return {"result": "success"}
```

### 3. Access the Dashboard

Once setup is complete, access the dashboard at:
```
http://localhost:5001
```

## Configuration

### Performance Configuration File

The system uses `performance_config.json` for configuration:

```json
{
  "database": {
    "path": "performance_metrics.db",
    "retention_hours": 168
  },
  "alerts": {
    "enabled": true,
    "rules": [
      {
        "component_id": "*",
        "metric_type": "latency",
        "threshold": 5000.0,
        "condition": "greater_than",
        "level": "warning",
        "description": "High latency detected"
      }
    ]
  },
  "dashboard": {
    "enabled": true,
    "host": "localhost",
    "port": 5001,
    "auto_refresh_seconds": 30
  },
  "monitoring": {
    "enabled": true,
    "auto_discover_components": true,
    "track_system_metrics": true,
    "track_agent_metrics": true,
    "track_tool_metrics": true
  },
  "notifications": {
    "console": true,
    "log": true,
    "email": false,
    "webhook": false
  }
}
```

### Alert Rules Configuration

Alert rules can be configured for different components and metrics:

```json
{
  "component_id": "research_agent",
  "metric_type": "latency",
  "threshold": 3000.0,
  "condition": "greater_than",
  "level": "warning",
  "description": "Research agent taking too long"
}
```

**Supported Conditions:**
- `greater_than`: Trigger when value > threshold
- `less_than`: Trigger when value < threshold
- `equals`: Trigger when value ≈ threshold

**Alert Levels:**
- `info`: Informational alerts
- `warning`: Warning level issues
- `error`: Error conditions
- `critical`: Critical system issues

## Advanced Usage

### Custom Metrics

Record custom metrics for your specific use cases:

```python
from core.performance_decorators import get_performance_monitor
from core.performance_monitor import MetricType

monitor = get_performance_monitor()
if monitor:
    # Record custom throughput
    monitor.collector.record_metric(
        component_id="my_component",
        component_type=ComponentType.AGENT,
        metric_type=MetricType.THROUGHPUT,
        value=15.5,
        unit="requests_per_second",
        tags={"operation": "data_processing"},
        metadata={"batch_size": 100}
    )
```

### Context Managers

Use context managers for fine-grained tracking:

```python
from core.performance_decorators import track_api_call, track_database_operation

# Track API calls
with track_api_call("my_service", "openai_api"):
    response = openai.chat.completions.create(...)

# Track database operations
with track_database_operation("user_service", "query_users"):
    users = database.query("SELECT * FROM users")
```

### Middleware Integration

For more complex integration scenarios:

```python
from core.performance_decorators import PerformanceMiddleware
from core.performance_monitor import ComponentType

class MyService:
    def __init__(self):
        self.perf = PerformanceMiddleware("my_service", ComponentType.SYSTEM)
    
    def complex_operation(self, data):
        with self.perf.track_execution("complex_operation"):
            # Your operation logic
            result = self.process_data(data)
            
            # Record custom metrics
            self.perf.record_metric(
                MetricType.RESOURCE_USAGE, 
                75.0, 
                "percent",
                tags={"resource": "memory"}
            )
            
            return result
```

### Integration with Existing Classes

Add monitoring to existing classes without modification:

```python
from core.performance_decorators import add_performance_monitoring_to_class

# Add monitoring to an existing agent class
MonitoredExistingAgent = add_performance_monitoring_to_class(
    ExistingAgentClass, 
    "existing_agent", 
    ComponentType.AGENT
)

# Use the monitored version
agent = MonitoredExistingAgent()
```

## Dashboard Features

### System Overview
- **Health Score**: Overall system health (0-100)
- **Total Executions**: Number of operations performed
- **Success Rate**: Percentage of successful operations
- **Average Latency**: Mean response time across all components
- **Active Alerts**: Current unresolved alerts
- **Component Count**: Number of monitored components

### Component Analysis
- **Individual Statistics**: Per-component performance metrics
- **Trend Analysis**: Performance trends over time
- **Bottleneck Detection**: Automatic identification of slow components
- **Resource Usage**: Memory and CPU utilization tracking

### Alert Management
- **Real-time Alerts**: Live alert notifications
- **Alert History**: Historical alert data
- **Acknowledge/Resolve**: Alert lifecycle management
- **Severity Filtering**: Filter alerts by severity level

### Performance Charts
- **Latency Trends**: Response time over time
- **Success Rate Trends**: Success percentage over time
- **Throughput Charts**: Request rate visualization
- **Resource Utilization**: System resource usage

## API Reference

### Core Classes

#### PerformanceMonitor
Main coordinator for the monitoring system.

```python
monitor = PerformanceMonitor(db_path="metrics.db")
monitor.start()  # Start monitoring
monitor.stop()   # Stop monitoring
health_score = monitor.get_system_health_score()
```

#### PerformanceCollector
Collects and stores performance metrics.

```python
execution_id = collector.start_execution(component_id, component_type, operation)
collector.end_execution(execution_id, success=True)
collector.record_metric(component_id, component_type, metric_type, value, unit)
```

#### AlertManager
Manages alerts and notifications.

```python
alert_manager.add_alert_rule(component_id, metric_type, threshold, condition, level)
alert_manager.acknowledge_alert(alert_id)
alert_manager.resolve_alert(alert_id)
```

#### PerformanceDashboard
Generates reports and dashboard data.

```python
overview = dashboard.generate_system_overview(time_window_hours=24)
component_report = dashboard.generate_component_report(component_id)
```

### Decorators

#### @monitor_agent_execution
Monitor agent method execution.

```python
@monitor_agent_execution("agent_id", "operation_name")
def agent_method(self, data):
    return process_data(data)
```

#### @monitor_tool_execution
Monitor tool method execution.

```python
@monitor_tool_execution("tool_id", "operation_name")
def tool_method(self, input_data):
    return transform_data(input_data)
```

#### @monitor_workflow_execution
Monitor workflow execution.

```python
@monitor_workflow_execution("workflow_id", "step_name")
def workflow_step(self, context):
    return execute_step(context)
```

## Best Practices

### 1. Component Naming
Use consistent, descriptive names for components:
```python
# Good
@monitor_agent_execution("research_agent", "web_search")
@monitor_tool_execution("data_processor", "clean_data")

# Avoid
@monitor_agent_execution("agent1", "func1")
```

### 2. Granular Monitoring
Monitor at appropriate granularity levels:
```python
# Monitor main operations
@monitor_agent_execution("nlp_agent", "analyze_sentiment")
def analyze_sentiment(self, text):
    # Don't monitor every small helper function
    tokens = self._tokenize(text)  # Not monitored
    features = self._extract_features(tokens)  # Not monitored
    return self._classify(features)  # Not monitored
```

### 3. Custom Metrics
Use custom metrics for domain-specific measurements:
```python
def process_documents(self, documents):
    with self.perf.track_execution("process_documents"):
        processed = []
        for doc in documents:
            result = self.process_single_doc(doc)
            processed.append(result)
        
        # Record custom metrics
        self.perf.record_metric(
            MetricType.THROUGHPUT,
            len(documents) / execution_time,
            "documents_per_second"
        )
        
        return processed
```

### 4. Alert Thresholds
Set realistic alert thresholds based on your system's normal behavior:
```json
{
  "component_id": "heavy_computation_agent",
  "metric_type": "latency",
  "threshold": 30000.0,
  "condition": "greater_than",
  "level": "warning",
  "description": "Heavy computation taking longer than expected"
}
```

### 5. Resource Monitoring
Monitor resource usage for resource-intensive operations:
```python
import psutil

def resource_intensive_operation(self, data):
    with self.perf.track_execution("intensive_operation"):
        # Monitor memory before operation
        memory_before = psutil.virtual_memory().percent
        
        result = self.heavy_computation(data)
        
        # Monitor memory after operation
        memory_after = psutil.virtual_memory().percent
        memory_used = memory_after - memory_before
        
        self.perf.record_metric(
            MetricType.MEMORY_USAGE,
            memory_used,
            "percent"
        )
        
        return result
```

## Troubleshooting

### Common Issues

#### 1. Dashboard Not Loading
- Check if the dashboard server is running
- Verify the port is not in use by another application
- Check firewall settings

#### 2. No Metrics Appearing
- Ensure monitoring is initialized: `initialize_performance_monitoring()`
- Verify decorators are applied correctly
- Check database permissions

#### 3. High Memory Usage
- Adjust retention hours in configuration
- Implement metric sampling for high-frequency operations
- Monitor database size

#### 4. Missing Dependencies
```bash
pip install flask plotly psutil numpy
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run dashboard in debug mode
python setup_performance_monitoring.py --dashboard --debug
```

### Performance Impact

The monitoring system is designed to have minimal performance impact:
- Metrics collection: < 1ms overhead per operation
- Database writes: Asynchronous, non-blocking
- Memory usage: Configurable retention limits
- CPU usage: < 1% additional load

## Examples

See the `examples/` directory for complete integration examples:
- `monitored_agent_example.py`: Agent integration examples
- `monitored_tool_example.py`: Tool integration examples

## Support

For issues and questions:
1. Check this documentation
2. Review the configuration file
3. Enable debug logging
4. Check the examples directory

## Future Enhancements

Planned features for future releases:
- **Distributed Monitoring**: Multi-node performance tracking
- **Machine Learning**: Anomaly detection and predictive alerts
- **Advanced Visualizations**: More chart types and analysis tools
- **Integration APIs**: REST API for external monitoring tools
- **Performance Optimization**: Automated performance tuning suggestions
- **Custom Dashboards**: User-configurable dashboard layouts