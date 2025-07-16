# Enhanced Hermes Documentation

## Overview

Enhanced Hermes is an advanced version of the AgentK orchestrator with three key improvements:

1. **Context Awareness** - Cross-session memory and learning capabilities
2. **Dynamic Plan Adaptation** - Real-time plan modification based on execution feedback  
3. **Multi-Tasking** - Priority-based task scheduling with dynamic reprioritization

## Key Features

### 🧠 Context Awareness

Enhanced Hermes maintains context across sessions, learning from past interactions to provide more intelligent orchestration.

**Capabilities:**
- **Cross-Session Memory**: Remembers user preferences, successful patterns, and failed approaches
- **User Preference Learning**: Adapts to user working styles and agent preferences
- **Pattern Recognition**: Identifies successful workflows and reuses them
- **Performance Tracking**: Monitors agent performance for better future assignments

**Implementation:**
- `ContextManager` class handles context storage and retrieval
- Integration with memory manager for persistent storage
- Context-aware agent recommendations based on historical performance

### 🔄 Dynamic Plan Adaptation

Plans can be modified in real-time based on execution feedback, reducing the need for user intervention when things go wrong.

**Capabilities:**
- **Failure Detection**: Automatically detects when plan steps fail
- **Adaptation Strategies**: Multiple strategies for plan modification
- **Real-Time Modification**: Plans adapt without stopping execution
- **Learning from Failures**: Adaptation history improves future planning

**Adaptation Strategies:**
- **Agent Substitution**: Replace unavailable agents with alternatives
- **Step Decomposition**: Break complex steps into smaller sub-tasks
- **Dependency Reordering**: Adjust execution order based on actual dependencies
- **Step Modification**: Modify approach based on error analysis
- **Parallel Execution**: Execute independent steps simultaneously
- **Fallback Strategy**: Use alternative approaches when primary fails

### 📋 Multi-Tasking

Manage multiple simultaneous tasks with intelligent priority-based scheduling.

**Capabilities:**
- **Priority-Based Scheduling**: Tasks executed based on priority and dependencies
- **Dynamic Reprioritization**: Priorities adjust based on urgency and interdependencies
- **Dependency Management**: Automatic handling of task dependencies
- **Resource Allocation**: Intelligent assignment of agents to tasks
- **Progress Tracking**: Real-time monitoring of task execution

**Priority Levels:**
- **CRITICAL**: Immediate execution required
- **HIGH**: Important tasks that should be prioritized
- **NORMAL**: Standard priority tasks
- **LOW**: Tasks that can be deferred

## Architecture

### Enhanced State Structure

```python
class EnhancedHermesState(TypedDict):
    # Standard Hermes state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan_step_count: int
    initial_goal: Optional[str]
    
    # Context awareness
    session_id: str
    user_id: Optional[str]
    context: Dict[str, Any]
    
    # Multi-tasking
    active_tasks: Dict[str, Task]
    task_queue: List[str]
    current_task_id: Optional[str]
    
    # Plan adaptation
    current_plan: Optional[ExecutionPlan]
    plan_history: List[str]
    adaptation_count: int
    
    # Performance tracking
    agent_performance: Dict[str, Dict[str, Any]]
    success_metrics: Dict[str, float]
```

### Core Components

#### TaskScheduler
Manages task creation, prioritization, and execution order.

```python
scheduler = TaskScheduler()
scheduler.add_task(task)
next_task = scheduler.get_next_task()
scheduler.mark_completed(task_id, result)
```

#### PlanAdapter
Handles dynamic plan adaptation based on execution feedback.

```python
adapter = PlanAdapter()
analysis = adapter.analyze_execution_feedback(plan, failed_step, error)
adapted_plan = adapter.adapt_plan(plan, adaptation_strategy)
```

#### ContextManager
Manages context awareness and cross-session learning.

```python
context_mgr = ContextManager()
context = context_mgr.load_context(session_id, user_id)
context_mgr.save_context_snapshot(session_id, goal, completed_tasks, insights)
```

## Enhanced Tools

### Task Management Tools

#### create_task
Create tasks with priority and dependency support.

```python
create_task(
    description="Implement user authentication",
    priority="HIGH",
    dependencies=["task_1"],
    assigned_agent="software_engineer",
    estimated_duration=120
)
```

#### update_task
Update task status, priority, or results.

```python
update_task(
    task_id="task_123",
    status="completed",
    result="Authentication implemented successfully"
)
```

#### list_tasks
List tasks with filtering options.

```python
list_tasks(
    status_filter="pending",
    priority_filter="HIGH",
    agent_filter="software_engineer"
)
```

#### get_next_task
Get the next task to execute based on priority and dependencies.

```python
get_next_task()  # Returns highest priority executable task
```

### Context Management Tools

#### save_context
Save context information for future sessions.

```python
save_context(
    session_id="session_123",
    context_type="preference",
    data={"preferred_agent": "software_engineer"},
    importance=8
)
```

#### load_context
Load context from previous sessions.

```python
load_context(
    session_id="session_123",
    context_type="preference",
    limit=10
)
```

#### update_user_preference
Update user preferences for future sessions.

```python
update_user_preference(
    preference_key="communication_style",
    preference_value="detailed_explanations",
    session_id="session_123"
)
```

### Plan Adaptation Tools

#### create_adaptive_plan
Create plans that can be modified based on execution feedback.

```python
create_adaptive_plan(
    goal="Build web application",
    constraints=["2 week deadline"],
    available_agents=["software_engineer", "web_researcher"],
    priority="HIGH"
)
```

#### adapt_plan
Adapt plans based on execution failures.

```python
adapt_plan(
    original_plan="Step 1: Research\nStep 2: Implement",
    failure_reason="Agent timeout",
    failed_step="Step 2: Implement",
    adaptation_strategy="agent_substitution"
)
```

#### analyze_plan_execution
Analyze plan execution and provide recommendations.

```python
analyze_plan_execution(
    plan_text="Original plan text",
    execution_feedback=[...],
    goal="Original goal"
)
```

## Usage Examples

### Basic Enhanced Session

```python
from agents.hermes_enhanced import enhanced_hermes

# Start enhanced session with context awareness
result = enhanced_hermes(
    uuid="session_123",
    user_id="user_456"  # Optional for cross-session context
)
```

### Context-Aware Workflow

```python
# Load previous context
context = load_context(session_id="session_123", user_id="user_456")

# Create tasks based on context
if context.get("preferred_agent") == "software_engineer":
    create_task(
        description="Code development task",
        assigned_agent="software_engineer",
        priority="HIGH"
    )

# Save insights for future sessions
save_context(
    session_id="session_123",
    context_type="insight",
    data={"successful_pattern": "step_by_step_development"},
    importance=8
)
```

### Multi-Task Management

```python
# Create multiple prioritized tasks
tasks = [
    {"description": "Fix security bug", "priority": "CRITICAL"},
    {"description": "Add new feature", "priority": "HIGH"},
    {"description": "Update docs", "priority": "LOW"}
]

for task_data in tasks:
    create_task(**task_data)

# Get next task based on priority
next_task = get_next_task()

# Execute and update
update_task(
    task_id=next_task["task"]["id"],
    status="completed",
    result="Task completed successfully"
)
```

### Plan Adaptation Workflow

```python
# Create adaptive plan
plan = create_adaptive_plan(
    goal="Complete project milestone",
    available_agents=["software_engineer", "web_researcher"]
)

# Simulate execution with failure
execution_feedback = [
    {"status": "success", "step_index": 0},
    {"status": "failure", "step_index": 1, "message": "Agent timeout"}
]

# Analyze and adapt
analysis = analyze_plan_execution(
    plan_text=plan["plan"]["original_plan"],
    execution_feedback=execution_feedback,
    goal=plan["plan"]["goal"]
)

if analysis["adaptation_recommendations"]:
    adapted_plan = adapt_plan(
        original_plan=plan["plan"]["original_plan"],
        failure_reason="Agent timeout",
        failed_step="Step 2",
        adaptation_strategy="agent_substitution"
    )
```

## Configuration

### Environment Variables

```bash
# Enhanced features
ENHANCED_HERMES_ENABLED=true
CONTEXT_RETENTION_DAYS=30
MAX_CONCURRENT_TASKS=10
ADAPTATION_THRESHOLD=3

# Memory management
MEMORY_MANAGER_ENABLED=true
CONTEXT_IMPORTANCE_THRESHOLD=5
```

### Initialization Options

```python
# Configure enhanced features
enhanced_hermes_config = {
    "context_awareness": True,
    "plan_adaptation": True,
    "multi_tasking": True,
    "max_tasks": 20,
    "adaptation_threshold": 3,
    "context_retention_days": 30
}
```

## Performance Metrics

Enhanced Hermes tracks several performance metrics:

### Context Awareness Metrics
- **Context Hit Rate**: Percentage of sessions that benefit from previous context
- **Preference Accuracy**: How well learned preferences match user behavior
- **Pattern Success Rate**: Success rate of reused patterns

### Plan Adaptation Metrics
- **Adaptation Success Rate**: Percentage of successful plan adaptations
- **Failure Recovery Time**: Time to recover from plan failures
- **Adaptation Frequency**: How often plans need adaptation

### Multi-Tasking Metrics
- **Task Completion Rate**: Percentage of tasks completed successfully
- **Priority Adherence**: How well the system follows priority ordering
- **Resource Utilization**: Efficiency of agent assignment

## Best Practices

### Context Management
1. **Regular Context Cleanup**: Remove old, irrelevant context data
2. **Importance Scoring**: Use appropriate importance levels for context items
3. **Privacy Considerations**: Be mindful of sensitive information in context

### Plan Adaptation
1. **Failure Analysis**: Thoroughly analyze failures before adapting
2. **Adaptation Limits**: Set reasonable limits on adaptation attempts
3. **Learning Integration**: Use adaptation history to improve future planning

### Multi-Tasking
1. **Priority Setting**: Use appropriate priority levels for tasks
2. **Dependency Management**: Clearly define task dependencies
3. **Resource Allocation**: Consider agent capabilities when assigning tasks

## Troubleshooting

### Common Issues

#### Context Not Loading
- Check memory manager initialization
- Verify session ID consistency
- Ensure context importance meets threshold

#### Plan Adaptation Failing
- Verify adaptation strategy is valid
- Check execution feedback format
- Ensure plan structure is correct

#### Task Scheduling Issues
- Verify task priorities are valid
- Check dependency relationships
- Ensure agents are available

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enhanced Hermes will provide detailed logs
enhanced_hermes(uuid="debug_session", user_id="debug_user")
```

## Future Enhancements

### Planned Features
1. **Advanced Learning**: Machine learning for better context utilization
2. **Predictive Adaptation**: Predict failures before they occur
3. **Resource Optimization**: Advanced resource allocation algorithms
4. **Cross-User Learning**: Learn from patterns across multiple users
5. **Real-Time Collaboration**: Multiple users working on shared goals

### Integration Opportunities
1. **External Calendars**: Integration with scheduling systems
2. **Project Management**: Integration with PM tools
3. **Communication Platforms**: Integration with Slack, Teams, etc.
4. **Monitoring Systems**: Integration with system monitoring tools

## Conclusion

Enhanced Hermes represents a significant advancement in AI orchestration, providing:

- **Intelligent Context Awareness** for personalized experiences
- **Dynamic Adaptation** for robust execution
- **Efficient Multi-Tasking** for complex workflows

These improvements make AgentK more capable, reliable, and user-friendly, enabling it to handle increasingly complex tasks with minimal user intervention while learning and improving over time.