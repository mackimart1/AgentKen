# AgentKen User Feedback Loop System

## 🎯 Overview

The AgentKen User Feedback Loop System enables users to rate task outcomes and implements continuous learning to improve agent and tool behaviors. The system automatically collects feedback, analyzes patterns, and applies machine learning techniques to enhance performance over time.

## 🌟 Key Features

### 📊 **Comprehensive Feedback Collection**
- **Multiple Feedback Types**: Rating (1-5 stars), Binary (thumbs up/down), Text, and Detailed aspect-based feedback
- **Automatic Tracking**: Seamless integration with existing agents and tools
- **Smart Prompting**: Intelligent feedback requests based on task importance and user behavior
- **User Context**: Session and user ID tracking for personalized insights

### 🧠 **Intelligent Analysis & Learning**
- **Performance Profiling**: Component-level performance analysis with trends
- **Pattern Recognition**: Identifies correlations between execution metrics and user satisfaction
- **Learning Insights**: Generates actionable insights from feedback patterns
- **Continuous Adaptation**: Automatically applies learning strategies to improve performance

### 🖥️ **Web Dashboard & Interface**
- **Real-time Dashboard**: Live feedback statistics and component performance
- **Interactive Feedback Forms**: User-friendly feedback collection interface
- **Performance Visualization**: Charts and graphs showing trends and insights
- **Alert Management**: Notifications for performance issues and learning opportunities

### 🔧 **Easy Integration**
- **Decorators**: Simple function decoration for automatic tracking
- **Base Classes**: Inherit from feedback-enabled agent/tool classes
- **Context Managers**: Fine-grained execution tracking
- **Manual Collection**: Programmatic feedback collection APIs

## 🚀 Quick Start

### 1. Setup and Installation

```bash
# Setup the feedback system
python setup_feedback_system.py

# Run with web interface
python setup_feedback_system.py --web

# Run the test suite
python test_feedback_system.py
```

### 2. Basic Integration

#### Using Decorators

```python
from core.feedback_integration import track_agent_execution, track_tool_execution

# Track agent execution
@track_agent_execution("research_agent", "web_search")
def search_web(query: str, user_id: str = None, session_id: str = None) -> dict:
    # Your agent logic here
    return {"results": ["result1", "result2"]}

# Track tool execution
@track_tool_execution("data_processor", "transform")
def transform_data(data: list, user_id: str = None, session_id: str = None) -> list:
    # Your tool logic here
    return [item.upper() for item in data]
```

#### Using Base Classes

```python
from core.feedback_integration import FeedbackEnabledAgent, FeedbackEnabledTool

class MyAgent(FeedbackEnabledAgent):
    def __init__(self):
        super().__init__("my_agent")
    
    def process_task(self, task_data: dict, user_id: str = None, session_id: str = None) -> dict:
        return self.execute_with_feedback(
            operation="process_task",
            func=self._do_process,
            user_id=user_id,
            session_id=session_id,
            input_data=task_data,
            task_data
        )
    
    def _do_process(self, task_data: dict) -> dict:
        # Your processing logic
        return {"status": "completed"}

class MyTool(FeedbackEnabledTool):
    def __init__(self):
        super().__init__("my_tool")
    
    def execute_operation(self, data: dict, user_id: str = None, session_id: str = None) -> dict:
        return self.execute_with_feedback(
            operation="execute",
            func=self._do_execute,
            user_id=user_id,
            session_id=session_id,
            input_data=data,
            data
        )
    
    def _do_execute(self, data: dict) -> dict:
        # Your tool logic
        return {"result": "success"}
```

### 3. Manual Feedback Collection

```python
from core.feedback_integration import (
    collect_rating_feedback, collect_binary_feedback, 
    collect_text_feedback, collect_detailed_feedback
)

# Rating feedback (1-5 stars)
feedback = collect_rating_feedback(
    task_execution_id="task_123",
    user_id="user_456",
    rating=4.5,
    text_feedback="Great results!",
    tags=["helpful", "accurate"]
)

# Binary feedback (thumbs up/down)
feedback = collect_binary_feedback(
    task_execution_id="task_123",
    user_id="user_456",
    satisfied=True,
    text_feedback="Worked perfectly!"
)

# Detailed aspect-based feedback
feedback = collect_detailed_feedback(
    task_execution_id="task_123",
    user_id="user_456",
    detailed_scores={
        "accuracy": 4.5,
        "speed": 4.0,
        "usefulness": 5.0
    },
    text_feedback="Excellent accuracy, could be faster"
)
```

### 4. Access the Web Interface

Once setup is complete, access the feedback dashboard at:
```
http://localhost:5002
```

## 📋 Configuration

### Feedback Configuration File

The system uses `feedback_config.json` for configuration:

```json
{
  "database": {
    "path": "feedback_system.db",
    "retention_days": 365
  },
  "collection": {
    "enabled": true,
    "auto_prompt": true,
    "prompt_strategy": "smart",
    "feedback_rate_target": 0.3
  },
  "learning": {
    "enabled": true,
    "auto_apply": true,
    "confidence_threshold": 0.7,
    "impact_threshold": 5.0
  },
  "web_interface": {
    "enabled": true,
    "host": "localhost",
    "port": 5002
  },
  "prompting": {
    "strategies": {
      "smart": {
        "enabled": true,
        "min_execution_time": 1.0,
        "failure_always_prompt": true,
        "success_prompt_rate": 0.3
      }
    }
  }
}
```

### Prompting Strategies

**Available Strategies:**
- `immediate`: Prompt immediately after task completion
- `delayed`: Prompt after a specified delay
- `batch`: Prompt when user has multiple completed tasks
- `smart`: Intelligent prompting based on task importance and user behavior

**Smart Strategy Rules:**
- Always prompt for failures and errors
- Prompt for long-running tasks (> 30 seconds)
- Skip very quick tasks (< 1 second)
- Respect user preferences
- Random sampling for regular tasks (30% by default)

## 🔍 Advanced Usage

### Custom Learning Strategies

```python
def custom_learning_strategy(insight):
    """Custom learning strategy based on feedback insights"""
    if insight.insight_type == "performance_correlation":
        correlation = insight.supporting_data.get("correlation", 0)
        if correlation < -0.7:  # Strong negative correlation with execution time
            return {
                "type": "performance_optimization",
                "component_id": insight.component_id,
                "adjustments": {
                    "timeout_reduction": 0.8,
                    "caching_enabled": True,
                    "async_processing": True
                },
                "reason": "Optimize speed based on user feedback"
            }
    return None

# Register the strategy
feedback_system.learning_engine.register_learning_strategy(
    "custom_optimization", custom_learning_strategy
)
```

### Performance Analysis

```python
# Get component performance profile
profile = feedback_system.analyzer.analyze_component_performance("my_agent")
print(f"Average Rating: {profile.average_rating:.2f}")
print(f"Satisfaction Rate: {profile.satisfaction_rate:.1f}%")
print(f"Improvement Trend: {profile.improvement_trend:.1f}%")
print(f"Strengths: {profile.strengths}")
print(f"Weaknesses: {profile.weaknesses}")
print(f"Recommendations: {profile.recommendations}")

# Generate learning insights
insights = feedback_system.analyzer.generate_learning_insights()
for insight in insights:
    print(f"Insight: {insight.description}")
    print(f"Confidence: {insight.confidence:.2f}")
    print(f"Impact: {insight.impact_score:.1f}/10")
```

### Context Managers for Fine-Grained Tracking

```python
from core.feedback_integration import FeedbackTracker, ComponentType

def complex_operation(data, user_id=None, session_id=None):
    with FeedbackTracker(
        component_id="complex_processor",
        component_type=ComponentType.TOOL,
        operation="complex_operation",
        user_id=user_id,
        session_id=session_id,
        input_data={"data_size": len(data)}
    ) as tracker:
        
        # Your complex operation logic
        result = process_data(data)
        
        # Set output data for feedback context
        tracker.set_output_data({"result_size": len(result)})
        
        return result
```

## 📊 Dashboard Features

### System Overview
- **Feedback Statistics**: Total feedback count, average ratings, satisfaction rates
- **Component Performance**: Individual agent/tool performance metrics
- **Learning Insights**: AI-generated insights from feedback analysis
- **Trend Analysis**: Performance trends over time

### Feedback Collection Interface
- **Multiple Input Types**: Rating stars, thumbs up/down, text areas, detailed scoring
- **Smart Forms**: Context-aware feedback forms based on task type
- **User-Friendly Design**: Intuitive interface for quick feedback submission
- **Mobile Responsive**: Works on desktop and mobile devices

### Learning Dashboard
- **Insight Visualization**: Display of learning insights with confidence scores
- **Adaptation History**: Track of applied learning strategies
- **Performance Impact**: Before/after analysis of learning adaptations
- **Manual Controls**: Ability to manually apply or reject learning suggestions

## 🔧 API Reference

### Core Classes

#### FeedbackSystem
Main coordinator for the feedback system.

```python
feedback_system = initialize_feedback_system(db_path="feedback.db")
feedback_system.set_prompt_strategy("smart")
feedback_system.enable_auto_prompt(True)
```

#### FeedbackCollector
Collects and manages user feedback.

```python
execution_id = collector.register_task_execution(execution)
feedback = collector.collect_feedback(task_id, user_id, feedback_type, **data)
pending_tasks = collector.get_pending_tasks_for_user(user_id)
```

#### FeedbackAnalyzer
Analyzes feedback to generate insights.

```python
profile = analyzer.analyze_component_performance(component_id, hours=168)
insights = analyzer.generate_learning_insights(component_id)
```

#### ContinuousLearningEngine
Implements learning strategies based on feedback.

```python
learning_engine.register_learning_strategy("strategy_name", strategy_function)
adaptations = learning_engine.apply_learning(component_id)
history = learning_engine.get_adaptation_history()
```

### Decorators

#### @track_agent_execution
Track agent method execution with feedback.

```python
@track_agent_execution("agent_id", "operation_name")
def agent_method(self, data, user_id=None, session_id=None):
    return process_data(data)
```

#### @track_tool_execution
Track tool method execution with feedback.

```python
@track_tool_execution("tool_id", "operation_name")
def tool_method(self, input_data, user_id=None, session_id=None):
    return transform_data(input_data)
```

### Feedback Collection Functions

```python
# Rating feedback (1-5 scale)
collect_rating_feedback(task_id, user_id, rating, text_feedback=None, tags=None)

# Binary feedback (satisfied/not satisfied)
collect_binary_feedback(task_id, user_id, satisfied, text_feedback=None, tags=None)

# Text feedback
collect_text_feedback(task_id, user_id, text_feedback, tags=None)

# Detailed aspect-based feedback
collect_detailed_feedback(task_id, user_id, detailed_scores, text_feedback=None, tags=None)
```

## 🎯 Best Practices

### 1. User Context Management
Always provide user_id and session_id when possible:
```python
@track_agent_execution("my_agent")
def process_request(request_data, user_id=None, session_id=None):
    # Extract user context from request if not provided
    if not user_id and 'user_id' in request_data:
        user_id = request_data['user_id']
    
    # Your processing logic
    return result
```

### 2. Meaningful Component Names
Use descriptive, consistent component IDs:
```python
# Good
@track_agent_execution("content_research_agent", "web_search")
@track_tool_execution("data_cleaning_tool", "remove_duplicates")

# Avoid
@track_agent_execution("agent1", "func1")
```

### 3. Appropriate Feedback Prompting
Configure prompting strategies based on your use case:
```python
# For critical operations - always prompt
feedback_system.prompt_system.set_user_preferences("user123", {
    "feedback_frequency": "all"
})

# For background tasks - minimal prompting
feedback_system.prompt_system.set_user_preferences("user456", {
    "feedback_frequency": "minimal"
})
```

### 4. Learning Strategy Design
Design learning strategies that make meaningful improvements:
```python
def quality_improvement_strategy(insight):
    if insight.insight_type == "satisfaction_variability":
        # Only apply if we have enough data and confidence
        if (insight.confidence > 0.8 and 
            insight.supporting_data.get("sample_size", 0) > 10):
            return {
                "type": "quality_enhancement",
                "adjustments": {
                    "validation_steps": ["consistency_check", "quality_gate"],
                    "quality_threshold": 0.9
                }
            }
    return None
```

### 5. Performance Impact Monitoring
Monitor the impact of learning adaptations:
```python
# Before applying learning
before_profile = analyzer.analyze_component_performance("my_agent")

# Apply learning
adaptations = learning_engine.apply_learning("my_agent")

# Wait for new data
time.sleep(3600)  # 1 hour

# Check impact
after_profile = analyzer.analyze_component_performance("my_agent")
improvement = after_profile.average_rating - before_profile.average_rating
print(f"Rating improvement: {improvement:.2f}")
```

## 📈 Feedback Types and Use Cases

### 1. Rating Feedback (1-5 Stars)
**Best for**: General satisfaction measurement
**Use cases**: Overall task quality, user satisfaction surveys
```python
collect_rating_feedback(task_id, user_id, 4.5, "Great results!")
```

### 2. Binary Feedback (Thumbs Up/Down)
**Best for**: Quick satisfaction checks
**Use cases**: Simple yes/no questions, quick polls
```python
collect_binary_feedback(task_id, user_id, True, "Worked perfectly!")
```

### 3. Text Feedback
**Best for**: Detailed user comments
**Use cases**: Bug reports, feature requests, detailed feedback
```python
collect_text_feedback(task_id, user_id, "The results were accurate but took too long to generate.")
```

### 4. Detailed Feedback
**Best for**: Multi-aspect evaluation
**Use cases**: Comprehensive quality assessment, UX evaluation
```python
collect_detailed_feedback(task_id, user_id, {
    "accuracy": 4.5,
    "speed": 3.0,
    "usefulness": 5.0,
    "ease_of_use": 4.0
}, "Excellent accuracy and very useful, but could be faster.")
```

## 🔍 Learning Insights Types

### 1. Performance Correlation
Identifies relationships between execution metrics and user satisfaction.
```
"Strong negative correlation between execution time and user satisfaction (r=-0.78)"
```

### 2. Satisfaction Variability
Detects inconsistent user experiences.
```
"High variability in user satisfaction (std=1.8) suggests inconsistent quality"
```

### 3. Error Patterns
Analyzes failure rates and their impact on user satisfaction.
```
"High error rate detected (15% of executions) with low user satisfaction"
```

### 4. Usage Patterns
Identifies trends in user behavior and preferences.
```
"Users prefer shorter responses during peak hours"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. No Feedback Being Collected
- Check if feedback system is initialized: `initialize_feedback_system()`
- Verify decorators are applied correctly
- Ensure user_id is provided in function calls
- Check database permissions

#### 2. Web Interface Not Loading
- Verify the web interface is enabled in configuration
- Check if port 5002 is available
- Ensure templates directory exists
- Check Flask dependencies

#### 3. Learning Not Applied
- Verify learning is enabled in configuration
- Check confidence and impact thresholds
- Ensure sufficient feedback data exists
- Review learning strategy implementations

#### 4. Missing User Context
```python
# Ensure user context is passed through
@track_agent_execution("my_agent")
def my_function(data, user_id=None, session_id=None):
    # user_id and session_id will be automatically tracked
    return process(data)
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug mode
python setup_feedback_system.py --web --debug
```

## 📊 Performance Impact

The feedback system is designed for minimal overhead:
- **Execution Tracking**: < 2ms additional overhead per operation
- **Database Operations**: Asynchronous, non-blocking writes
- **Memory Usage**: Configurable retention with automatic cleanup
- **CPU Usage**: < 2% additional system load

## 🔮 Future Enhancements

Planned features for future releases:
- **Advanced ML Models**: Deep learning for pattern recognition
- **Predictive Analytics**: Predict user satisfaction before task completion
- **A/B Testing**: Built-in experimentation framework
- **Multi-modal Feedback**: Voice and image feedback support
- **Real-time Adaptation**: Instant learning application
- **Federated Learning**: Cross-system learning without data sharing

## 📚 Examples

See the `examples/` directory for complete integration examples:
- `feedback_agent_example.py`: Agent integration examples
- `feedback_tool_example.py`: Tool integration examples

## 🎉 Conclusion

The AgentKen User Feedback Loop System provides a comprehensive solution for collecting user feedback and implementing continuous learning. The system is designed to be:

- **Easy to integrate** with existing agents and tools
- **Intelligent** in feedback collection and analysis
- **Adaptive** through machine learning strategies
- **Scalable** for production environments
- **User-friendly** with intuitive interfaces

By implementing this feedback system, AgentKen can continuously improve its performance based on real user experiences, leading to better outcomes and higher user satisfaction over time.