"""
Task Tracker Agent: Manages and tracks tasks within the AgentK system.

This agent provides comprehensive task management capabilities including:
- Creating and organizing tasks
- Tracking task progress and status
- Managing task dependencies and priorities
- Generating task reports and analytics
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config

# Setup logger
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class Task:
    """Task data structure."""
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = None
    dependencies: List[str] = None
    progress: float = 0.0  # 0-100%
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "assigned_to": self.assigned_to,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "progress": self.progress
        }


class TaskTrackerAgent:
    """Task tracker class for managing tasks."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
    
    def create_task(self, title: str, description: str, priority: str = "normal", 
                   assigned_to: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """Create a new task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:04d}"
        
        # Convert priority string to enum
        try:
            priority_enum = TaskPriority[priority.upper()]
        except KeyError:
            priority_enum = TaskPriority.NORMAL
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority_enum,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            assigned_to=assigned_to,
            tags=tags or []
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id}: {title}")
        return task_id
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        if task_id not in self.tasks:
            return False
        
        try:
            status_enum = TaskStatus[status.upper()]
            self.tasks[task_id].status = status_enum
            self.tasks[task_id].updated_at = datetime.now()
            logger.info(f"Updated task {task_id} status to {status}")
            return True
        except KeyError:
            logger.error(f"Invalid status: {status}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def list_tasks(self, status: Optional[str] = None, assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        filtered_tasks = []
        
        for task in self.tasks.values():
            # Filter by status if specified
            if status and task.status.value != status.lower():
                continue
            
            # Filter by assignee if specified
            if assigned_to and task.assigned_to != assigned_to:
                continue
            
            filtered_tasks.append(task.to_dict())
        
        # Sort by priority (highest first) then by created date
        filtered_tasks.sort(key=lambda x: (-x["priority"], x["created_at"]))
        return filtered_tasks
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        summary = {
            "total_tasks": len(self.tasks),
            "by_status": {},
            "by_priority": {},
            "overdue_tasks": 0
        }
        
        # Count by status
        for status in TaskStatus:
            summary["by_status"][status.value] = 0
        
        # Count by priority
        for priority in TaskPriority:
            summary["by_priority"][priority.name.lower()] = 0
        
        now = datetime.now()
        for task in self.tasks.values():
            summary["by_status"][task.status.value] += 1
            summary["by_priority"][task.priority.name.lower()] += 1
            
            # Check for overdue tasks
            if task.due_date and task.due_date < now and task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
                summary["overdue_tasks"] += 1
        
        return summary


# Global task tracker instance
task_tracker = TaskTrackerAgent()

# System prompt for the agent
system_prompt = """You are task_tracker_agent, a specialized ReAct agent for comprehensive task management within the AgentK system.

Your capabilities include:
- Creating and organizing tasks with priorities and assignments
- Tracking task progress and status updates
- Managing task dependencies and relationships
- Generating task reports and analytics
- Providing task recommendations and insights

You can help users:
1. Create new tasks with detailed specifications
2. Update existing task status and progress
3. List and filter tasks by various criteria
4. Generate task summaries and reports
5. Manage task assignments and priorities

Always provide clear, actionable responses and maintain accurate task records.
Use the available tools to perform task management operations effectively.
"""

# Import required tools
from tools.scratchpad import scratchpad
from tools.list_available_agents import list_available_agents
from tools.assign_agent_to_task import assign_agent_to_task

tools = [scratchpad, list_available_agents, assign_agent_to_task]


def reasoning(state: MessagesState) -> dict:
    """Task tracker reasoning step."""
    print("task_tracker_agent is analyzing task management request...")
    messages = state["messages"]
    
    # Use Google Gemini for tool calling from hybrid configuration
    tool_model = config.get_model_for_tools()
    if tool_model is None:
        # Fallback to default model if hybrid setup fails
        tool_model = config.default_langchain_model
        logger.warning("Using fallback model for tools - may not support function calling")
    
    tooled_up_model = tool_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}


def check_for_tool_calls(state: MessagesState):
    """Check for tool calls in the last message."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if last_message.content and last_message.content.strip():
            print("task_tracker_agent thought this:")
            print(last_message.content)
        print()
        print("task_tracker_agent is taking action by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"

    return END


# Build workflow
acting = ToolNode(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges(
    "reasoning", 
    check_for_tool_calls,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "reasoning")

graph = workflow.compile()


def task_tracker_agent(task: str) -> Dict[str, Any]:
    """
    Task Tracker Agent for comprehensive task management.
    
    Args:
        task (str): The task management request or command
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
    """
    try:
        logger.info(f"Task Tracker Agent processing: {task}")
        
        # Handle direct task management commands
        task_lower = task.lower()
        
        if task_lower.startswith("create task"):
            # Extract task details from the request
            # This is a simplified parser - could be enhanced
            parts = task.split(":", 1)
            if len(parts) > 1:
                title = parts[1].strip()
                task_id = task_tracker.create_task(title, task)
                return {
                    "status": "success",
                    "result": task_id,
                    "message": f"Created task {task_id}: {title}"
                }
        
        elif task_lower.startswith("list tasks"):
            tasks = task_tracker.list_tasks()
            return {
                "status": "success",
                "result": tasks,
                "message": f"Retrieved {len(tasks)} tasks"
            }
        
        elif task_lower.startswith("task summary"):
            summary = task_tracker.get_task_summary()
            return {
                "status": "success",
                "result": summary,
                "message": "Generated task summary"
            }
        
        # For complex requests, use the LangGraph workflow
        final_state = graph.invoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ]
        })
        
        # Extract final message
        last_message_content = "Task management operation completed"
        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                content = last_message.content
                if isinstance(content, list):
                    last_message_content = next(
                        (item for item in content if isinstance(item, str)), 
                        str(content)
                    )
                else:
                    last_message_content = str(content)
        
        return {
            "status": "success",
            "result": last_message_content,
            "message": last_message_content
        }
        
    except Exception as e:
        error_msg = f"Task Tracker Agent error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "failure",
            "result": None,
            "message": error_msg
        }


# Export the main function
__all__ = ["task_tracker_agent", "TaskTrackerAgent", "Task", "TaskStatus", "TaskPriority"]