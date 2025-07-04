# Inferra V - Enhanced Agent System

## Overview

The Inferra V Enhanced Agent System is a sophisticated multi-agent orchestration platform that provides intelligent task planning, execution, and adaptation with real-time optimization and comprehensive monitoring.

## Key Features

### 🚀 Advanced Orchestration
- **Adaptive Planning**: Dynamic plan adaptation based on execution feedback
- **Intelligent Task Scheduling**: Respects dependencies and optimizes execution order
- **Resource Management**: Efficient allocation and monitoring of system resources
- **Parallel Execution**: Automatic identification and execution of parallelizable tasks

### 🤖 Agent Framework
- **Multi-Agent Coordination**: Seamless communication between specialized agents
- **Capability-Based Routing**: Automatic task assignment based on agent capabilities
- **Load Balancing**: Intelligent distribution of tasks across available agents
- **Health Monitoring**: Continuous monitoring of agent status and performance

### 🛠️ Tool Integration
- **Robust Error Handling**: Circuit breaker pattern and retry mechanisms
- **Rate Limiting**: Prevents system overload with configurable rate limits
- **Performance Monitoring**: Detailed metrics collection and analysis
- **Tool Registry**: Centralized management of available tools and capabilities

### 📊 Workflow Monitoring
- **Real-time Metrics**: System performance, resource utilization, and health scores
- **Bottleneck Detection**: Automatic identification of performance bottlenecks
- **Alert Management**: Configurable alerts with multiple severity levels
- **Optimization Recommendations**: AI-driven suggestions for system improvements

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main System   │    │  Core Modules   │    │   Monitoring    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Configuration │    │ • Orchestrator  │    │ • Metrics       │
│ • Initialization│    │ • Agent Framework│    │ • Alerts        │
│ • Coordination  │    │ • Tool Registry │    │ • Optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation
```bash
# Navigate to the project directory
cd "C:\Users\Macki Marinez\Desktop\2025 Projects\Inferra V"

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Configuration
The system uses environment variables for configuration. Key settings include:

```bash
# Model Configuration
DEFAULT_MODEL_PROVIDER=GOOGLE
DEFAULT_MODEL_NAME=models/gemini-1.5-flash-preview-0514
DEFAULT_MODEL_TEMPERATURE=0.1

# System Configuration
ORCHESTRATOR_MAX_CONCURRENT_TASKS=20
AGENT_HEARTBEAT_INTERVAL=30
TOOL_DEFAULT_TIMEOUT=30
LOG_LEVEL=INFO
```

### 3. Running the System
```bash
python main.py
```

## Core Components

### Adaptive Orchestrator (`core/adaptive_orchestrator.py`)
The heart of the system that manages task execution and planning:

- **Plan Creation**: Converts task definitions into executable plans
- **Optimization**: Applies multiple optimization strategies
- **Execution**: Manages task scheduling and execution
- **Adaptation**: Dynamically adjusts plans based on performance

### Agent Framework (`core/agent_framework.py`)
Manages agent communication and coordination:

- **Message Bus**: Central communication hub for all agents
- **Agent Registry**: Tracks available agents and their capabilities
- **Load Balancing**: Distributes tasks efficiently across agents

### Tool Integration (`core/tool_integration_system.py`)
Provides robust tool management:

- **Circuit Breaker**: Prevents cascading failures
- **Rate Limiting**: Controls tool usage rates
- **Error Recovery**: Automatic retry with exponential backoff
- **Performance Tracking**: Detailed execution metrics

### Workflow Monitoring (`core/workflow_monitoring.py`)
Comprehensive system monitoring:

- **Metrics Collection**: Real-time system and application metrics
- **Alert Management**: Configurable alerting system
- **Bottleneck Detection**: Automatic performance issue identification
- **Optimization Engine**: Generates improvement recommendations

## Usage Examples

### Creating a Custom Plan
```python
from core.adaptive_orchestrator import AdvancedOrchestrator, TaskPriority

# Define tasks
tasks = [
    {
        'id': 'task_1',
        'name': 'Data Collection',
        'agent_type': 'data_agent',
        'capability': 'collect_data',
        'parameters': {'source': 'api', 'format': 'json'},
        'priority': TaskPriority.HIGH.value,
        'estimated_duration': 10.0,
        'dependencies': [],
        'required_resources': ['network_access']
    },
    {
        'id': 'task_2',
        'name': 'Data Processing',
        'agent_type': 'processing_agent',
        'capability': 'process_data',
        'parameters': {'algorithm': 'ml_analysis'},
        'priority': TaskPriority.NORMAL.value,
        'estimated_duration': 15.0,
        'dependencies': ['task_1'],
        'required_resources': ['cpu_intensive']
    }
]

# Create and execute plan
plan_id = orchestrator.create_plan(
    name="Data Analysis Pipeline",
    description="Collect and process data for analysis",
    tasks=tasks,
    deadline=time.time() + 1800  # 30 minutes
)

orchestrator.execute_plan(plan_id)
```

### Adding Custom Tools
```python
from core.tool_integration_system import BaseTool, ToolDefinition

class CustomTool(BaseTool):
    def __init__(self):
        definition = ToolDefinition(
            name="custom_tool",
            version="1.0.0",
            description="Custom tool for specific tasks",
            input_schema={"input": "string"},
            output_schema={"result": "string"},
            timeout=30.0,
            max_retries=3
        )
        super().__init__(definition)
    
    def _execute(self, input: str) -> dict:
        # Custom tool logic here
        return {"result": f"Processed: {input}"}

# Register the tool
tool_registry.register_tool(CustomTool())
```

### Monitoring System Health
```python
# Get system status
status = orchestrator.get_system_metrics()
print(f"Active Plans: {status['active_plans']}")
print(f"Agent Utilization: {status['agent_utilization']}")

# Get optimization report
report = monitor.get_optimization_report()
print(f"Health Score: {report['system_health_score']}/100")
print(f"Recommendations: {len(report['optimization_recommendations'])}")
```

## Performance Metrics

The system tracks various performance metrics:

- **Execution Time**: Task and plan completion times
- **Resource Utilization**: CPU, memory, and custom resource usage
- **Error Rates**: Success/failure ratios for tasks and tools
- **Throughput**: Tasks completed per unit time
- **Latency**: Response times for various operations

## Alert Configuration

Alerts can be configured for various conditions:

```python
# Add custom alert
monitor.alert_manager.add_alert_rule(
    metric_name="custom_metric",
    threshold=100.0,
    condition="greater_than",
    severity=AlertSeverity.WARNING,
    description="Custom metric exceeded threshold"
)
```

## Troubleshooting

### Common Issues

1. **Tasks Not Executing**
   - Check agent capabilities match task requirements
   - Verify resource constraints are properly configured
   - Ensure task dependencies are correctly specified

2. **High Memory Usage**
   - Adjust memory alert thresholds in workflow monitoring
   - Review task execution patterns for memory leaks
   - Consider scaling resources or optimizing algorithms

3. **Performance Degradation**
   - Check bottleneck detection reports
   - Review optimization recommendations
   - Monitor resource utilization patterns

### Debugging

Enable debug logging for detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Machine Learning Integration**: Predictive optimization based on historical data
- **Distributed Execution**: Support for multi-node deployments
- **Advanced Visualization**: Real-time dashboards and performance graphs
- **API Gateway**: RESTful API for external system integration

## Contributing

To contribute to the enhanced system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility where possible

## License

This enhanced system builds upon the original Inferra V project and maintains the same licensing terms.