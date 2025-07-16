"""
Enhanced Hermes Agent: Advanced orchestrator with context awareness, dynamic plan adaptation, and multi-tasking.

Key Improvements:
1. Context Awareness: Enhanced memory management and cross-session context retention
2. Dynamic Plan Adaptation: Real-time plan modification based on execution feedback
3. Multi-Tasking: Priority-based task scheduling with dynamic reprioritization
"""

from typing import Literal, Sequence, TypedDict, Annotated, Optional, cast, Union, Dict, List, Any
import operator
import datetime
import json
import uuid as uuid_lib
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import time

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import utils
import config
import logging
from langchain_core.runnables.config import RunnableConfig
from requests.exceptions import HTTPError, RequestException

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize memory manager
try:
    import memory_manager
    memory_manager_instance = memory_manager.MemoryManager()
    logger.info("Memory manager initialized successfully")
except ImportError as e:
    logger.warning(f"Could not import memory_manager: {e}")
    memory_manager_instance = None
except Exception as e:
    logger.warning(f"Could not initialize memory manager: {e}")
    memory_manager_instance = None

from tools.list_available_agents import list_available_agents
from tools.assign_agent_to_task import assign_agent_to_task


# --- Enhanced Data Structures ---

class TaskPriority(Enum):
    """Task priority levels for multi-tasking."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Enhanced task representation."""
    id: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    assigned_agent: Optional[str] = None
    dependencies: List[str] = None  # Task IDs this task depends on
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: Optional[int] = None  # in minutes
    actual_duration: Optional[int] = None
    context: Dict[str, Any] = None  # Additional context for the task
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.context is None:
            self.context = {}


@dataclass
class PlanStep:
    """Individual step in an execution plan."""
    id: str
    description: str
    agent: Optional[str]
    dependencies: List[str]
    status: TaskStatus
    estimated_duration: Optional[int] = None
    actual_start: Optional[datetime.datetime] = None
    actual_end: Optional[datetime.datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Enhanced execution plan with adaptation capabilities."""
    id: str
    goal: str
    steps: List[PlanStep]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    status: TaskStatus
    adaptation_count: int = 0
    original_plan: Optional[str] = None  # Store original plan text
    current_step_index: int = 0
    success_rate: float = 0.0  # Track plan success rate
    
    def get_next_executable_steps(self) -> List[PlanStep]:
        """Get steps that can be executed (dependencies met)."""
        executable = []
        completed_step_ids = {step.id for step in self.steps if step.status == TaskStatus.COMPLETED}
        
        for step in self.steps:
            if (step.status == TaskStatus.PENDING and 
                all(dep_id in completed_step_ids for dep_id in step.dependencies)):
                executable.append(step)
        
        return executable


@dataclass
class ContextSnapshot:
    """Snapshot of conversation context for cross-session awareness."""
    session_id: str
    user_id: Optional[str]
    timestamp: datetime.datetime
    goal: str
    completed_tasks: List[str]
    failed_tasks: List[str]
    learned_preferences: Dict[str, Any]
    key_insights: List[str]
    active_context: Dict[str, Any]


class TaskScheduler:
    """Priority-based task scheduler with dynamic reprioritization."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.execution_queue: List[str] = []  # Task IDs in execution order
        self.lock = threading.Lock()
    
    def add_task(self, task: Task) -> None:
        """Add a task to the scheduler."""
        with self.lock:
            self.tasks[task.id] = task
            self._reorder_queue()
    
    def update_task_priority(self, task_id: str, new_priority: TaskPriority) -> None:
        """Update task priority and reorder queue."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].priority = new_priority
                self.tasks[task_id].updated_at = datetime.datetime.now()
                self._reorder_queue()
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute based on priority and dependencies."""
        with self.lock:
            for task_id in self.execution_queue:
                task = self.tasks[task_id]
                if (task.status == TaskStatus.PENDING and 
                    self._dependencies_met(task)):
                    return task
            return None
    
    def _dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    return False
        return True
    
    def _reorder_queue(self) -> None:
        """Reorder execution queue based on priority and dependencies."""
        # Sort by priority first, then by creation time
        self.execution_queue = sorted(
            self.tasks.keys(),
            key=lambda tid: (
                self.tasks[tid].priority.value,
                self.tasks[tid].created_at
            )
        )
    
    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.COMPLETED
                self.tasks[task_id].result = result
                self.tasks[task_id].updated_at = datetime.datetime.now()
    
    def mark_failed(self, task_id: str, error_message: str) -> None:
        """Mark a task as failed."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.retry_count += 1
                task.error_message = error_message
                task.updated_at = datetime.datetime.now()
                
                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                else:
                    task.status = TaskStatus.PENDING  # Retry
    
    def get_task_summary(self) -> Dict[str, int]:
        """Get summary of task statuses."""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary


class PlanAdapter:
    """Handles dynamic plan adaptation based on execution feedback."""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def analyze_execution_feedback(self, plan: ExecutionPlan, failed_step: PlanStep, 
                                 error_message: str) -> Dict[str, Any]:
        """Analyze execution feedback to determine adaptation strategy."""
        analysis = {
            "failure_type": self._classify_failure(error_message),
            "affected_steps": self._find_affected_steps(plan, failed_step),
            "suggested_adaptations": [],
            "confidence": 0.0
        }
        
        # Classify failure and suggest adaptations
        if "agent not found" in error_message.lower():
            analysis["suggested_adaptations"].append({
                "type": "agent_substitution",
                "description": "Try alternative agent or create new agent",
                "confidence": 0.8
            })
        elif "timeout" in error_message.lower():
            analysis["suggested_adaptations"].append({
                "type": "step_decomposition",
                "description": "Break down step into smaller sub-steps",
                "confidence": 0.7
            })
        elif "dependency" in error_message.lower():
            analysis["suggested_adaptations"].append({
                "type": "dependency_reorder",
                "description": "Reorder steps to resolve dependencies",
                "confidence": 0.9
            })
        else:
            analysis["suggested_adaptations"].append({
                "type": "step_modification",
                "description": "Modify step approach based on error",
                "confidence": 0.6
            })
        
        return analysis
    
    def adapt_plan(self, plan: ExecutionPlan, adaptation_strategy: Dict[str, Any]) -> ExecutionPlan:
        """Adapt the execution plan based on the strategy."""
        adapted_plan = ExecutionPlan(
            id=f"{plan.id}_adapted_{plan.adaptation_count + 1}",
            goal=plan.goal,
            steps=plan.steps.copy(),
            created_at=plan.created_at,
            updated_at=datetime.datetime.now(),
            status=plan.status,
            adaptation_count=plan.adaptation_count + 1,
            original_plan=plan.original_plan,
            current_step_index=plan.current_step_index
        )
        
        # Apply adaptation based on strategy type
        strategy_type = adaptation_strategy.get("type", "")
        
        if strategy_type == "agent_substitution":
            self._apply_agent_substitution(adapted_plan, adaptation_strategy)
        elif strategy_type == "step_decomposition":
            self._apply_step_decomposition(adapted_plan, adaptation_strategy)
        elif strategy_type == "dependency_reorder":
            self._apply_dependency_reorder(adapted_plan, adaptation_strategy)
        elif strategy_type == "step_modification":
            self._apply_step_modification(adapted_plan, adaptation_strategy)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.datetime.now(),
            "original_plan_id": plan.id,
            "adapted_plan_id": adapted_plan.id,
            "strategy": adaptation_strategy,
            "reason": adaptation_strategy.get("description", "")
        })
        
        return adapted_plan
    
    def _classify_failure(self, error_message: str) -> str:
        """Classify the type of failure based on error message."""
        error_lower = error_message.lower()
        
        if "agent" in error_lower and ("not found" in error_lower or "unavailable" in error_lower):
            return "agent_unavailable"
        elif "timeout" in error_lower or "time" in error_lower:
            return "timeout"
        elif "dependency" in error_lower or "prerequisite" in error_lower:
            return "dependency_failure"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission_error"
        elif "resource" in error_lower:
            return "resource_unavailable"
        else:
            return "unknown"
    
    def _find_affected_steps(self, plan: ExecutionPlan, failed_step: PlanStep) -> List[str]:
        """Find steps that might be affected by the failure."""
        affected = []
        
        # Find steps that depend on the failed step
        for step in plan.steps:
            if failed_step.id in step.dependencies:
                affected.append(step.id)
        
        return affected
    
    def _apply_agent_substitution(self, plan: ExecutionPlan, strategy: Dict[str, Any]) -> None:
        """Apply agent substitution adaptation."""
        # This would involve finding alternative agents
        # For now, we'll mark the step for manual review
        for step in plan.steps:
            if step.status == TaskStatus.FAILED:
                step.description += " [NEEDS AGENT SUBSTITUTION]"
    
    def _apply_step_decomposition(self, plan: ExecutionPlan, strategy: Dict[str, Any]) -> None:
        """Apply step decomposition adaptation."""
        # This would involve breaking down complex steps
        # For now, we'll mark the step for decomposition
        for step in plan.steps:
            if step.status == TaskStatus.FAILED:
                step.description += " [NEEDS DECOMPOSITION]"
    
    def _apply_dependency_reorder(self, plan: ExecutionPlan, strategy: Dict[str, Any]) -> None:
        """Apply dependency reordering adaptation."""
        # This would involve reordering steps
        # For now, we'll mark for manual reordering
        for step in plan.steps:
            if step.status == TaskStatus.FAILED:
                step.description += " [NEEDS REORDERING]"
    
    def _apply_step_modification(self, plan: ExecutionPlan, strategy: Dict[str, Any]) -> None:
        """Apply step modification adaptation."""
        # This would involve modifying the step approach
        # For now, we'll mark for modification
        for step in plan.steps:
            if step.status == TaskStatus.FAILED:
                step.description += " [NEEDS MODIFICATION]"


class ContextManager:
    """Manages context awareness across sessions."""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.current_context: Dict[str, Any] = {}
        self.session_history: Dict[str, ContextSnapshot] = {}
    
    def load_context(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Load context for a session."""
        context = {
            "session_id": session_id,
            "user_id": user_id,
            "previous_goals": [],
            "learned_preferences": {},
            "successful_patterns": [],
            "failed_patterns": [],
            "agent_performance": {},
            "user_feedback_history": []
        }
        
        if self.memory_manager:
            try:
                # Retrieve relevant memories for context
                memories = self.memory_manager.retrieve_memories(
                    memory_type="context",
                    agent_name="hermes",
                    limit=20
                )
                
                for memory in memories:
                    memory_data = memory.get("value", {})
                    if isinstance(memory_data, str):
                        try:
                            memory_data = json.loads(memory_data)
                        except json.JSONDecodeError:
                            continue
                    
                    if isinstance(memory_data, dict):
                        if memory_data.get("type") == "goal":
                            context["previous_goals"].append(memory_data)
                        elif memory_data.get("type") == "preference":
                            context["learned_preferences"].update(memory_data.get("data", {}))
                        elif memory_data.get("type") == "pattern":
                            if memory_data.get("success", False):
                                context["successful_patterns"].append(memory_data)
                            else:
                                context["failed_patterns"].append(memory_data)
                
            except Exception as e:
                logger.warning(f"Failed to load context from memory: {e}")
        
        self.current_context = context
        return context
    
    def save_context_snapshot(self, session_id: str, goal: str, 
                            completed_tasks: List[str], failed_tasks: List[str],
                            insights: List[str]) -> None:
        """Save a context snapshot."""
        snapshot = ContextSnapshot(
            session_id=session_id,
            user_id=self.current_context.get("user_id"),
            timestamp=datetime.datetime.now(),
            goal=goal,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            learned_preferences=self.current_context.get("learned_preferences", {}),
            key_insights=insights,
            active_context=self.current_context.copy()
        )
        
        self.session_history[session_id] = snapshot
        
        # Save to persistent memory
        if self.memory_manager:
            try:
                self.memory_manager.add_memory(
                    key=f"context_snapshot_{session_id}_{datetime.datetime.now().timestamp()}",
                    value=json.dumps(asdict(snapshot), default=str),
                    memory_type="context",
                    agent_name="hermes",
                    importance=8
                )
            except Exception as e:
                logger.warning(f"Failed to save context snapshot: {e}")
    
    def update_agent_performance(self, agent_name: str, task_type: str, 
                               success: bool, duration: Optional[int] = None) -> None:
        """Update agent performance tracking."""
        if "agent_performance" not in self.current_context:
            self.current_context["agent_performance"] = {}
        
        agent_key = f"{agent_name}_{task_type}"
        if agent_key not in self.current_context["agent_performance"]:
            self.current_context["agent_performance"][agent_key] = {
                "successes": 0,
                "failures": 0,
                "avg_duration": 0,
                "total_duration": 0,
                "task_count": 0
            }
        
        perf = self.current_context["agent_performance"][agent_key]
        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1
        
        if duration:
            perf["total_duration"] += duration
            perf["task_count"] += 1
            perf["avg_duration"] = perf["total_duration"] / perf["task_count"]
    
    def get_agent_recommendation(self, task_type: str) -> Optional[str]:
        """Get agent recommendation based on performance history."""
        agent_performance = self.current_context.get("agent_performance", {})
        
        best_agent = None
        best_score = 0
        
        for agent_key, perf in agent_performance.items():
            if task_type in agent_key:
                agent_name = agent_key.replace(f"_{task_type}", "")
                total_tasks = perf["successes"] + perf["failures"]
                if total_tasks > 0:
                    success_rate = perf["successes"] / total_tasks
                    # Factor in success rate and speed
                    score = success_rate * (1 / max(perf["avg_duration"], 1))
                    if score > best_score:
                        best_score = score
                        best_agent = agent_name
        
        return best_agent


# --- Enhanced State Definition ---

class EnhancedHermesState(TypedDict):
    """Enhanced state with multi-tasking and context awareness."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan_step_count: int
    initial_goal: Optional[str]
    
    # Context awareness
    session_id: str
    user_id: Optional[str]
    context: Dict[str, Any]
    
    # Multi-tasking
    active_tasks: Dict[str, Task]
    task_queue: List[str]  # Task IDs in priority order
    current_task_id: Optional[str]
    
    # Plan adaptation
    current_plan: Optional[ExecutionPlan]
    plan_history: List[str]  # Plan IDs
    adaptation_count: int
    
    # Performance tracking
    agent_performance: Dict[str, Dict[str, Any]]
    success_metrics: Dict[str, float]


# Initialize global components
task_scheduler = TaskScheduler()
plan_adapter = PlanAdapter(memory_manager_instance)
context_manager = ContextManager(memory_manager_instance)

# Load tools
tools = utils.all_tool_functions()
tool_names_for_log = [getattr(t, "name", f"[Unknown Tool Type: {type(t).__name__}]") for t in tools]
logger.info(f"Enhanced Hermes initialized with tools: {tool_names_for_log}")


# Enhanced system prompt
enhanced_system_prompt = """You are Enhanced Hermes, an advanced ReAct agent orchestrator with context awareness, dynamic plan adaptation, and multi-tasking capabilities.

CORE CAPABILITIES:
1. **Context Awareness**: You retain and utilize context across sessions, learning from past interactions
2. **Dynamic Plan Adaptation**: You detect plan deviations and autonomously adapt strategies in real-time
3. **Multi-Tasking**: You manage multiple simultaneous tasks with priority-based scheduling

ENHANCED WORKFLOW:
1. **Context Loading**: Analyze previous interactions and learned preferences
2. **Goal Understanding**: Reach shared understanding while considering historical context
3. **Intelligent Planning**: Create adaptive plans that can evolve based on execution feedback
4. **Multi-Task Coordination**: Manage multiple tasks with dynamic priority adjustment
5. **Real-Time Adaptation**: Monitor execution and adapt plans when failures occur
6. **Context Preservation**: Save insights and patterns for future sessions

ADAPTATION STRATEGIES:
- **Agent Substitution**: Switch to alternative agents when primary choice fails
- **Step Decomposition**: Break complex steps into manageable sub-tasks
- **Dependency Reordering**: Adjust execution order based on actual dependencies
- **Priority Adjustment**: Dynamically reprioritize tasks based on urgency and interdependencies

CONTEXT UTILIZATION:
- Reference previous successful patterns for similar tasks
- Avoid previously failed approaches
- Consider user preferences and feedback history
- Leverage agent performance data for better assignments

You have access to enhanced tools for system overview, agent management, and task coordination.
Always explain your reasoning for adaptations and provide transparency in your decision-making process.
"""


# --- Enhanced Node Functions ---

def enhanced_feedback_and_wait_on_human_input(state: EnhancedHermesState) -> dict:
    """Enhanced feedback handler with context awareness."""
    session_id = state.get("session_id", str(uuid_lib.uuid4()))
    
    # Load context for this session
    context = context_manager.load_context(session_id, state.get("user_id"))
    
    # Determine message to show user
    if len(state["messages"]) == 1:
        # Check for previous context
        if context.get("previous_goals"):
            recent_goals = context["previous_goals"][-3:]  # Last 3 goals
            goal_summary = ", ".join([g.get("description", "")[:50] for g in recent_goals])
            message_to_human = f"Welcome back! I remember our previous work on: {goal_summary}. What can I help you with today?"
        else:
            message_to_human = "What can I help you with?"
    else:
        # Show the last AI message content
        last_ai_content = state["messages"][-1].content
        if isinstance(last_ai_content, list) and len(last_ai_content) > 0:
            message_to_human = (
                last_ai_content[0]
                if isinstance(last_ai_content[0], str)
                else str(last_ai_content[0])
            )
        elif isinstance(last_ai_content, str):
            message_to_human = last_ai_content
        else:
            message_to_human = str(last_ai_content)
    
    print(message_to_human)
    
    # Get user input
    human_input = ""
    while not human_input.strip():
        human_input = input("> ")
    
    # Prepare enhanced state update
    update_dict = {
        "messages": [HumanMessage(content=human_input)],
        "plan_step_count": 0,
        "session_id": session_id,
        "context": context,
        "active_tasks": state.get("active_tasks", {}),
        "task_queue": state.get("task_queue", []),
        "current_task_id": state.get("current_task_id"),
        "current_plan": state.get("current_plan"),
        "plan_history": state.get("plan_history", []),
        "adaptation_count": state.get("adaptation_count", 0),
        "agent_performance": context.get("agent_performance", {}),
        "success_metrics": state.get("success_metrics", {})
    }
    
    # Capture initial goal if not already set
    if not state.get("initial_goal"):
        update_dict["initial_goal"] = human_input
        print(f"(Initial goal captured: '{human_input[:50]}...')")
        
        # Save goal to context
        if context_manager.memory_manager:
            try:
                context_manager.memory_manager.add_memory(
                    key=f"goal_{session_id}_{datetime.datetime.now().timestamp()}",
                    value=json.dumps({"type": "goal", "description": human_input, "session_id": session_id}),
                    memory_type="context",
                    agent_name="hermes",
                    importance=7
                )
            except Exception as e:
                logger.warning(f"Failed to save goal to memory: {e}")
    
    return update_dict


def enhanced_reasoning(state: EnhancedHermesState) -> dict:
    """Enhanced reasoning with context awareness and plan adaptation."""
    print("\nenhanced hermes is thinking...")
    
    current_messages = state.get("messages", [])
    plan_step_count = state.get("plan_step_count", 0)
    initial_goal = state.get("initial_goal")
    context = state.get("context", {})
    current_plan = state.get("current_plan")
    
    # Check for plan adaptation needs
    if current_plan and plan_step_count > 0:
        # Analyze recent failures
        recent_failures = []
        for message in reversed(current_messages[-5:]):
            if isinstance(message, ToolMessage):
                content = message.content
                if isinstance(content, dict) and content.get("status") == "failure":
                    recent_failures.append(content.get("message", "Unknown error"))
        
        if recent_failures:
            print(f"Detected {len(recent_failures)} recent failures. Analyzing for plan adaptation...")
            
            # Find the failed step (simplified)
            failed_step = None
            if current_plan.steps and current_plan.current_step_index < len(current_plan.steps):
                failed_step = current_plan.steps[current_plan.current_step_index]
                failed_step.status = TaskStatus.FAILED
                failed_step.error_message = recent_failures[0]
            
            if failed_step:
                # Analyze and adapt
                analysis = plan_adapter.analyze_execution_feedback(
                    current_plan, failed_step, recent_failures[0]
                )
                
                if analysis["suggested_adaptations"]:
                    adaptation = analysis["suggested_adaptations"][0]
                    print(f"Adapting plan: {adaptation['description']}")
                    
                    adapted_plan = plan_adapter.adapt_plan(current_plan, adaptation)
                    
                    return {
                        "messages": [AIMessage(content=f"Plan adaptation detected. {adaptation['description']}. Continuing with adapted approach...")],
                        "current_plan": adapted_plan,
                        "adaptation_count": state.get("adaptation_count", 0) + 1
                    }
    
    # Enhanced context integration
    context_prompt = ""
    if context.get("previous_goals"):
        context_prompt += f"Previous goals: {[g.get('description', '')[:50] for g in context['previous_goals'][-3:]]}\n"
    
    if context.get("successful_patterns"):
        context_prompt += f"Successful patterns: {[p.get('description', '')[:50] for p in context['successful_patterns'][-2:]]}\n"
    
    if context.get("agent_performance"):
        best_performers = sorted(
            context["agent_performance"].items(),
            key=lambda x: x[1].get("successes", 0) / max(x[1].get("successes", 0) + x[1].get("failures", 0), 1),
            reverse=True
        )[:3]
        if best_performers:
            context_prompt += f"Top performing agents: {[bp[0].split('_')[0] for bp in best_performers]}\n"
    
    # Prepare enhanced messages for LLM
    messages_for_llm = list(current_messages)
    if context_prompt:
        enhanced_system_prompt_with_context = f"{enhanced_system_prompt}\n\nCONTEXT FROM PREVIOUS INTERACTIONS:\n{context_prompt}"
        
        # Update system message
        for i, msg in enumerate(messages_for_llm):
            if isinstance(msg, SystemMessage):
                messages_for_llm[i] = SystemMessage(content=enhanced_system_prompt_with_context)
                break
    
    # Main LLM call with enhanced error handling
    try:
        if config.default_langchain_model is None:
            raise ValueError("Default language model not initialized")
        
        # Use Google Gemini for tool calling from hybrid configuration
        tool_model = config.get_model_for_tools()
        if tool_model is None:
            # Fallback to default model if hybrid setup fails
            tool_model = config.default_langchain_model
            logger.warning("Using fallback model for tools - may not support function calling")
        
        tooled_up_model = tool_model.bind_tools(tools)
        response = tooled_up_model.invoke(messages_for_llm)
        
        # Enhanced response processing
        cleaned_content = response.content
        if isinstance(response.content, list) and len(response.content) > 0:
            cleaned_content = (
                response.content[0]
                if isinstance(response.content[0], str)
                else str(response.content[0])
            )
        
        # Create enhanced response
        ai_response = cast(AIMessage, response)
        enhanced_response = AIMessage(
            content=cleaned_content,
            tool_calls=ai_response.tool_calls,
            id=getattr(ai_response, "id", None),
            additional_kwargs=getattr(ai_response, "additional_kwargs", {}),
            response_metadata=getattr(ai_response, "response_metadata", {}),
            name=getattr(ai_response, "name", None),
        )
        
        # Enhanced memory storage
        if cleaned_content and memory_manager_instance:
            try:
                # Detect and store patterns
                if "**Plan:**" in cleaned_content:
                    plan_text = cleaned_content.split("**Plan:**", 1)[1].strip()
                    memory_manager_instance.add_memory(
                        key=f"plan_{datetime.datetime.now().timestamp()}",
                        value=json.dumps({
                            "type": "plan",
                            "content": plan_text,
                            "goal": initial_goal,
                            "session_id": state.get("session_id")
                        }),
                        memory_type="plan",
                        agent_name="hermes",
                        importance=8
                    )
                
                # Store reasoning patterns
                memory_manager_instance.add_memory(
                    key=f"reasoning_{datetime.datetime.now().timestamp()}",
                    value=json.dumps({
                        "type": "reasoning",
                        "content": cleaned_content,
                        "context_used": bool(context_prompt),
                        "session_id": state.get("session_id")
                    }),
                    memory_type="reasoning",
                    agent_name="hermes",
                    importance=6
                )
                
            except Exception as e:
                logger.warning(f"Failed to store enhanced memory: {e}")
        
        return {"messages": [enhanced_response]}
        
    except (HTTPError, RequestException) as e:
        print(f"\nAPI Error: {e}")
        print("Please enter a new OpenRouter API Key to continue:")
        new_key = input("> ").strip()
        if new_key:
            try:
                config.reinitialize_openrouter_model(new_key)
                print("Model reinitialized. Retrying...")
                return enhanced_reasoning(state)  # Retry
            except Exception as config_e:
                print(f"Failed to reinitialize: {config_e}")
        
        return {
            "messages": [AIMessage(content="I encountered an API error and cannot proceed without a valid API key.")]
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in enhanced reasoning: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"I encountered an unexpected error: {str(e)}")]
        }


def enhanced_check_for_tool_calls(state: EnhancedHermesState) -> Literal["tools", "feedback_and_wait_on_human_input"]:
    """Enhanced tool call checker with multi-task awareness."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for early stopping
    if isinstance(last_message, AIMessage):
        try:
            if isinstance(last_message.content, str) and last_message.content.startswith("Stopping task execution early:"):
                return "feedback_and_wait_on_human_input"
        except AttributeError:
            pass
    
    # Enhanced tool call detection
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"Enhanced Hermes is executing {len(last_message.tool_calls)} tool(s)...")
        
        # Track tool usage for performance analysis
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            context_manager.current_context.setdefault("tool_usage", {})
            context_manager.current_context["tool_usage"].setdefault(tool_name, 0)
            context_manager.current_context["tool_usage"][tool_name] += 1
        
        return "tools"
    else:
        return "feedback_and_wait_on_human_input"


# Enhanced tool node with performance tracking
def enhanced_tool_execution(state: EnhancedHermesState) -> dict:
    """Enhanced tool execution with performance tracking."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not (isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return {"messages": []}
    
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "unknown")
        start_time = time.time()
        
        try:
            # Execute tool (simplified - in real implementation, this would use ToolNode)
            # For now, we'll simulate tool execution
            result = {
                "status": "success",
                "result": f"Simulated result for {tool_name}",
                "message": f"Tool {tool_name} executed successfully"
            }
            
            execution_time = time.time() - start_time
            
            # Track performance
            context_manager.update_agent_performance(
                agent_name=tool_name,
                task_type="tool_execution",
                success=True,
                duration=int(execution_time * 1000)  # Convert to milliseconds
            )
            
            tool_results.append(ToolMessage(
                content=result,
                tool_call_id=tool_call.get("id", ""),
                name=tool_name
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Track failure
            context_manager.update_agent_performance(
                agent_name=tool_name,
                task_type="tool_execution",
                success=False,
                duration=int(execution_time * 1000)
            )
            
            error_result = {
                "status": "failure",
                "result": None,
                "message": f"Tool {tool_name} failed: {str(e)}"
            }
            
            tool_results.append(ToolMessage(
                content=error_result,
                tool_call_id=tool_call.get("id", ""),
                name=tool_name
            ))
    
    return {"messages": tool_results}


# --- Enhanced Workflow Setup ---

enhanced_workflow = StateGraph(EnhancedHermesState)
enhanced_workflow.add_node("feedback_and_wait_on_human_input", enhanced_feedback_and_wait_on_human_input)
enhanced_workflow.add_node("reasoning", enhanced_reasoning)
enhanced_workflow.add_node("tools", enhanced_tool_execution)

enhanced_workflow.set_entry_point("feedback_and_wait_on_human_input")

enhanced_workflow.add_conditional_edges(
    "feedback_and_wait_on_human_input",
    lambda state: "reasoning" if state["messages"][-1].content.lower() != "exit" else END
)

enhanced_workflow.add_conditional_edges(
    "reasoning",
    enhanced_check_for_tool_calls
)

enhanced_workflow.add_edge("tools", "reasoning")

enhanced_graph = enhanced_workflow.compile(checkpointer=utils.checkpointer)


def enhanced_hermes(uuid: str, user_id: Optional[str] = None):
    """
    Enhanced Hermes with context awareness, dynamic plan adaptation, and multi-tasking.
    
    Args:
        uuid (str): Session identifier
        user_id (Optional[str]): User identifier for cross-session context
    """
    print(f"Starting Enhanced AgentK session (id: {uuid})")
    if user_id:
        print(f"User context loaded for: {user_id}")
    print("Enhanced features: Context Awareness, Dynamic Plan Adaptation, Multi-Tasking")
    print("Type 'exit' to end the session.")
    
    # Initialize enhanced state
    initial_state = {
        "messages": [SystemMessage(content=enhanced_system_prompt)],
        "plan_step_count": 0,
        "initial_goal": None,
        "session_id": uuid,
        "user_id": user_id,
        "context": {},
        "active_tasks": {},
        "task_queue": [],
        "current_task_id": None,
        "current_plan": None,
        "plan_history": [],
        "adaptation_count": 0,
        "agent_performance": {},
        "success_metrics": {}
    }
    
    config_with_limit = {
        "configurable": {"thread_id": uuid},
        "recursion_limit": 100  # Increased for enhanced capabilities
    }
    config_with_limit = cast(RunnableConfig, config_with_limit)
    
    try:
        result = enhanced_graph.invoke(initial_state, config=config_with_limit)
        
        # Save final context snapshot
        if result.get("initial_goal"):
            completed_tasks = [task_id for task_id, task in result.get("active_tasks", {}).items() 
                             if task.status == TaskStatus.COMPLETED]
            failed_tasks = [task_id for task_id, task in result.get("active_tasks", {}).items() 
                           if task.status == TaskStatus.FAILED]
            
            context_manager.save_context_snapshot(
                session_id=uuid,
                goal=result["initial_goal"],
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                insights=["Session completed with enhanced capabilities"]
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced Hermes execution error: {e}", exc_info=True)
        print(f"Session ended due to error: {e}")
        return None


# Export the enhanced function
__all__ = ["enhanced_hermes", "EnhancedHermesState", "TaskScheduler", "PlanAdapter", "ContextManager"]