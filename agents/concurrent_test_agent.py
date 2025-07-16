from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def concurrent_test_agent(task: List[str]) -> Dict[str, Any]:
    """
    Executes multiple tasks concurrently and returns aggregated results.

    Args:
        task (List[str]): A list of tasks to execute concurrently.

    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message.
    """
    try:
        results = {}

        def execute_task(t: str) -> str:
            """Helper function to execute a single task."""
            try:
                # Simulate task execution (replace with actual logic)
                return f"Task '{t}' completed successfully"
            except Exception as e:
                logger.error(f"Error executing task '{t}': {e}")
                return f"Task '{t}' failed: {str(e)}"

        # Execute tasks concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(execute_task, t): t for t in task}
            for future in as_completed(futures):
                task_name = futures[future]
                results[task_name] = future.result()

        return {
            "status": "success",
            "result": results,
            "message": f"Successfully executed {len(task)} tasks concurrently",
        }
    except Exception as e:
        logger.error(f"Error in concurrent_test_agent: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to execute tasks concurrently: {str(e)}",
        }
