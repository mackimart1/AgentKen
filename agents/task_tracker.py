
import logging
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta

# Assuming the tools are in a 'tools' directory and are properly structured
from tools.list_tasks import list_tasks
from tools.assign_agent_to_task import assign_agent_to_task

logger = logging.getLogger(__name__)

def task_tracker(task: str) -> Dict[str, Any]:
    """
    Monitors the status of ongoing tasks, identifies overdue ones,
    and alerts the 'hermes' agent.

    Args:
        task (str): The task description, not directly used but required by the agent format.

    Returns:
        Dict[str, Any]: A dictionary with the status, result, and a message.
    """
    logger.info(f"Starting TaskTracker agent with task: {task}")
    overdue_tasks = []
    try:
        # 1. List all ongoing tasks
        # We filter for 'IN_PROGRESS' or 'PENDING' statuses.
        tasks_result_str = list_tasks(status_filter='IN_PROGRESS')
        
        # The tool returns a JSON string, so we need to parse it.
        tasks_result = json.loads(tasks_result_str)
        
        if tasks_result.get('status') != 'success':
            error_message = f"Failed to list tasks: {tasks_result.get('message')}"
            logger.error(error_message)
            return {"status": "failure", "result": None, "message": error_message}

        tasks = tasks_result.get('tasks', [])
        now = datetime.utcnow()

        # 2. Identify overdue tasks
        for t in tasks:
            created_at_str = t.get('created_at')
            estimated_duration_minutes = t.get('estimated_duration')

            if created_at_str and estimated_duration_minutes is not None:
                try:
                    # Assuming created_at is in ISO 8601 format (e.g., '2023-10-27T10:00:00Z')
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    
                    # Calculate the expected completion time
                    expected_completion = created_at + timedelta(minutes=estimated_duration_minutes)

                    if now > expected_completion:
                        overdue_tasks.append(t)
                        logger.warning(f"Task {t['id']} is overdue.")

                except (ValueError, TypeError) as e:
                    logger.error(f"Could not parse date or duration for task {t.get('id')}: {e}")
                    continue

        if not overdue_tasks:
            success_message = "No overdue tasks found."
            logger.info(success_message)
            return {"status": "success", "result": [], "message": success_message}

        # 3. Alert 'hermes' agent for each overdue task
        alerts_sent = []
        for overdue_task in overdue_tasks:
            alert_task_description = (
                f"ALERT: Task '{overdue_task['id']}' has exceeded its estimated completion time. "
                f"Description: '{overdue_task['description']}'. Please investigate."
            )
            try:
                logger.info(f"Assigning alert to hermes for task: {overdue_task['id']}")
                assignment_result_str = assign_agent_to_task(
                    agent_name='hermes',
                    task=alert_task_description
                )
                assignment_result = json.loads(assignment_result_str)

                if assignment_result.get('status') == 'success':
                    alerts_sent.append(overdue_task['id'])
                    logger.info(f"Successfully assigned alert for task {overdue_task['id']} to hermes.")
                else:
                    logger.error(f"Failed to assign alert for task {overdue_task['id']} to hermes: {assignment_result.get('message')}")

            except Exception as e:
                logger.error(f"Error assigning task to hermes: {e}")

        final_message = f"TaskTracker finished. Overdue tasks found: {len(overdue_tasks)}. Alerts sent for tasks: {alerts_sent}"
        return {
            "status": "success",
            "result": {"overdue_count": len(overdue_tasks), "alerts_sent_for": alerts_sent},
            "message": final_message,
        }

    except Exception as e:
        error_msg = f"An unexpected error occurred in TaskTracker: {e}"
        logger.critical(error_msg, exc_info=True)
        return {"status": "failure", "result": None, "message": error_msg}
