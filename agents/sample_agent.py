"""
Sample agent for versioning demonstration.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def sample_agent(task: str) -> Dict[str, Any]:
    """
    Sample agent for demonstration purposes.
    
    Args:
        task (str): The task to perform
        
    Returns:
        Dict[str, Any]: Result dictionary
    """
    try:
        logger.info(f"Sample agent executing task: {task}")
        
        # Simple task processing
        result = f"Processed: {task}"
        
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully completed task: {task}"
        }
        
    except Exception as e:
        logger.error(f"Error in sample_agent: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to complete task: {str(e)}"
        }
