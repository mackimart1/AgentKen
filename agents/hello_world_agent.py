from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def hello_world_agent(task: str) -> Dict[str, Any]:
    """
    A simple agent that returns a greeting message.

    Args:
        task (str): The task to perform. This is ignored by the agent,
                    but included for compatibility.

    Returns:
        Dict[str, Any]: A dictionary with the status, result, and a message.
    """
    logger.info(f"Executing hello_world_agent for task: {task}")
    try:
        greeting = "Hello, World!"
        return {
            "status": "success",
            "result": greeting,
            "message": "Successfully generated a greeting.",
        }
    except Exception as e:
        logger.error(f"Error in hello_world_agent: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to generate greeting: {str(e)}",
        }
