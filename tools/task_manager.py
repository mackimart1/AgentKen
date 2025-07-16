from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import datetime
from enum import Enum


class TaskPriority(Enum):
    """Task priority levels."""
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


class CreateTaskInput(BaseModel):
    description: str = Field(description="Task description")
    priority: str = Field(default="NORMAL", description="Task priority: CRITICAL, HIGH, NORMAL, LOW")
    dependencies: List[str] = Field(default=[], description="List of task IDs this task depends on")
    assigned_agent: Optional[str] = Field(default=None, description="Agent to assign this task to")
    estimated_duration: Optional[int] = Field(default=None, description="Estimated duration in minutes")
    context: Dict[str, Any] = Field(default={}, description="Additional context for the task")


class UpdateTaskInput(BaseModel):
    task_id: str = Field(description="Task ID to update")
    status: Optional[str] = Field(default=None, description="New task status")
    priority: Optional[str] = Field(default=None, description="New task priority")
    result: Optional[str] = Field(default=None, description="Task result")
    error_message: Optional[str] = Field(default=None, description="Error message if task failed")


class ListTasksInput(BaseModel):
    status_filter: Optional[str] = Field(default=None, description="Filter by status")
    priority_filter: Optional[str] = Field(default=None, description="Filter by priority")
    agent_filter: Optional[str] = Field(default=None, description="Filter by assigned agent")
    include_completed: bool = Field(default=True, description="Include completed tasks")


# Global task storage (in production, this would be a database)
_task_storage: Dict[str, Dict[str, Any]] = {}
_task_counter = 0


def _generate_task_id() -> str:
    """Generate a unique task ID."""
    global _task_counter
    _task_counter += 1
    return f"task_{_task_counter}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


@tool(args_schema=CreateTaskInput)
def create_task(
    description: str,
    priority: str = "NORMAL",
    dependencies: List[str] = None,
    assigned_agent: Optional[str] = None,
    estimated_duration: Optional[int] = None,
    context: Dict[str, Any] = None
) -> str:
    """
    Create a new task in the task management system.
    
    Args:
        description: Task description
        priority: Task priority (CRITICAL, HIGH, NORMAL, LOW)
        dependencies: List of task IDs this task depends on
        assigned_agent: Agent to assign this task to
        estimated_duration: Estimated duration in minutes
        context: Additional context for the task
    
    Returns:
        JSON string with task creation result
    """
    try:
        if dependencies is None:
            dependencies = []
        if context is None:
            context = {}
        
        # Validate priority
        try:
            priority_enum = TaskPriority[priority.upper()]
        except KeyError:
            return json.dumps({
                "status": "failure",
                "message": f"Invalid priority: {priority}. Must be one of: CRITICAL, HIGH, NORMAL, LOW"
            })
        
        # Generate task ID
        task_id = _generate_task_id()
        
        # Create task
        task = {
            "id": task_id,
            "description": description,
            "priority": priority.upper(),
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "assigned_agent": assigned_agent,
            "dependencies": dependencies,
            "result": None,
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3,
            "estimated_duration": estimated_duration,
            "actual_duration": None,
            "context": context
        }
        
        # Store task
        _task_storage[task_id] = task
        
        return json.dumps({
            "status": "success",
            "task_id": task_id,
            "message": f"Task created successfully: {description[:50]}..."
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to create task: {str(e)}"
        })


@tool(args_schema=UpdateTaskInput)
def update_task(
    task_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    result: Optional[str] = None,
    error_message: Optional[str] = None
) -> str:
    """
    Update an existing task.
    
    Args:
        task_id: Task ID to update
        status: New task status
        priority: New task priority
        result: Task result
        error_message: Error message if task failed
    
    Returns:
        JSON string with update result
    """
    try:
        if task_id not in _task_storage:
            return json.dumps({
                "status": "failure",
                "message": f"Task not found: {task_id}"
            })
        
        task = _task_storage[task_id]
        
        # Update status
        if status:
            try:
                status_enum = TaskStatus(status.lower())
                task["status"] = status_enum.value
            except ValueError:
                return json.dumps({
                    "status": "failure",
                    "message": f"Invalid status: {status}"
                })
        
        # Update priority
        if priority:
            try:
                priority_enum = TaskPriority[priority.upper()]
                task["priority"] = priority.upper()
            except KeyError:
                return json.dumps({
                    "status": "failure",
                    "message": f"Invalid priority: {priority}"
                })
        
        # Update result and error
        if result:
            task["result"] = result
        if error_message:
            task["error_message"] = error_message
            task["retry_count"] = task.get("retry_count", 0) + 1
        
        task["updated_at"] = datetime.datetime.now().isoformat()
        
        return json.dumps({
            "status": "success",
            "message": f"Task {task_id} updated successfully"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to update task: {str(e)}"
        })


@tool(args_schema=ListTasksInput)
def list_tasks(
    status_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
    include_completed: bool = True
) -> str:
    """
    List tasks with optional filtering.
    
    Args:
        status_filter: Filter by status
        priority_filter: Filter by priority
        agent_filter: Filter by assigned agent
        include_completed: Include completed tasks
    
    Returns:
        JSON string with task list
    """
    try:
        filtered_tasks = []
        
        for task in _task_storage.values():
            # Apply filters
            if status_filter and task["status"] != status_filter.lower():
                continue
            if priority_filter and task["priority"] != priority_filter.upper():
                continue
            if agent_filter and task["assigned_agent"] != agent_filter:
                continue
            if not include_completed and task["status"] == TaskStatus.COMPLETED.value:
                continue
            
            filtered_tasks.append(task)
        
        # Sort by priority and creation time
        priority_order = {p.name: p.value for p in TaskPriority}
        filtered_tasks.sort(key=lambda t: (
            priority_order.get(t["priority"], 999),
            t["created_at"]
        ))
        
        return json.dumps({
            "status": "success",
            "tasks": filtered_tasks,
            "count": len(filtered_tasks),
            "message": f"Found {len(filtered_tasks)} tasks"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to list tasks: {str(e)}"
        })


@tool
def get_task_summary() -> str:
    """
    Get a summary of all tasks by status and priority.
    
    Returns:
        JSON string with task summary
    """
    try:
        summary = {
            "by_status": {},
            "by_priority": {},
            "total_tasks": len(_task_storage),
            "pending_high_priority": 0,
            "overdue_tasks": 0
        }
        
        # Count by status and priority
        for task in _task_storage.values():
            status = task["status"]
            priority = task["priority"]
            
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1
            
            # Count high priority pending tasks
            if status == TaskStatus.PENDING.value and priority in ["CRITICAL", "HIGH"]:
                summary["pending_high_priority"] += 1
        
        return json.dumps({
            "status": "success",
            "summary": summary,
            "message": "Task summary generated successfully"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to generate task summary: {str(e)}"
        })


@tool
def get_next_task() -> str:
    """
    Get the next task to execute based on priority and dependencies.
    
    Returns:
        JSON string with next task information
    """
    try:
        # Find executable tasks (pending with met dependencies)
        executable_tasks = []
        completed_task_ids = {
            task_id for task_id, task in _task_storage.items()
            if task["status"] == TaskStatus.COMPLETED.value
        }
        
        for task in _task_storage.values():
            if task["status"] == TaskStatus.PENDING.value:
                # Check if dependencies are met
                dependencies_met = all(
                    dep_id in completed_task_ids
                    for dep_id in task.get("dependencies", [])
                )
                if dependencies_met:
                    executable_tasks.append(task)
        
        if not executable_tasks:
            return json.dumps({
                "status": "success",
                "task": None,
                "message": "No executable tasks available"
            })
        
        # Sort by priority
        priority_order = {p.name: p.value for p in TaskPriority}
        executable_tasks.sort(key=lambda t: (
            priority_order.get(t["priority"], 999),
            t["created_at"]
        ))
        
        next_task = executable_tasks[0]
        
        return json.dumps({
            "status": "success",
            "task": next_task,
            "message": f"Next task: {next_task['description'][:50]}..."
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get next task: {str(e)}"
        })


@tool
def clear_completed_tasks() -> str:
    """
    Clear all completed tasks from the task storage.
    
    Returns:
        JSON string with operation result
    """
    try:
        completed_count = 0
        tasks_to_remove = []
        
        for task_id, task in _task_storage.items():
            if task["status"] == TaskStatus.COMPLETED.value:
                tasks_to_remove.append(task_id)
                completed_count += 1
        
        for task_id in tasks_to_remove:
            del _task_storage[task_id]
        
        return json.dumps({
            "status": "success",
            "cleared_count": completed_count,
            "message": f"Cleared {completed_count} completed tasks"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to clear completed tasks: {str(e)}"
        })