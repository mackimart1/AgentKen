import sys
import traceback
from langchain_core.tools import tool
import utils
import logging
import concurrent.futures  # Import concurrent.futures
import time  # Import time for potential delays if needed
import os  # Import os for path checking

logger = logging.getLogger(__name__)  # Setup logger

# Define a timeout duration in seconds
AGENT_EXECUTION_TIMEOUT = 900  # 15 minutes (increased for complex agent creation tasks)


@tool
def assign_agent_to_task(agent_name: str, task: str):
    """
    Assigns a task to a registered agent and returns its structured response.
    Uses the agents_manifest.json to find and load the agent.
    Enforces a timeout on agent execution.
    Verifies file creation for agent_smith and tool_maker upon success.
    """
    print(f"Attempting to assign agent '{agent_name}' to task: {task[:100]}...")
    logger.info(
        f"Assigning task to agent '{agent_name}'. Timeout: {AGENT_EXECUTION_TIMEOUT}s"
    )

    agent_details = utils.get_agent_details(agent_name)
    if not agent_details:
        error_message = f"Agent '{agent_name}' not found in the manifest."
        logger.error(error_message)
        return {"status": "failure", "result": None, "message": error_message}

    try:
        agent_function = utils.load_registered_module(agent_details)

        if agent_function is None:
            error_message = f"Failed to load agent function for '{agent_name}'."
            # Error already logged by load_registered_module
            return {"status": "failure", "result": None, "message": error_message}

        structured_result = None
        # Use ThreadPoolExecutor to run the agent function with a timeout
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"agent_exec_{agent_name}"
        ) as executor:
            future = executor.submit(agent_function, task=task)
            try:
                structured_result = future.result(timeout=AGENT_EXECUTION_TIMEOUT)
                logger.info(f"Agent '{agent_name}' completed within timeout.")
            except concurrent.futures.TimeoutError:
                error_message = f"Agent '{agent_name}' timed out after {AGENT_EXECUTION_TIMEOUT} seconds while executing task: {task[:100]}..."
                logger.error(error_message)
                cancelled = future.cancel()
                logger.warning(
                    f"Attempted to cancel timed-out agent '{agent_name}'. Cancel successful: {cancelled}"
                )
                return {"status": "failure", "result": None, "message": error_message}
            except Exception as agent_exec_e:
                exception_trace = traceback.format_exc()
                error_message = f"Agent '{agent_name}' raised an exception during execution:\n {agent_exec_e}\n{exception_trace}"
                logger.error(error_message)
                return {"status": "failure", "result": None, "message": error_message}

        # --- Post-execution validation ---
        if structured_result is None:
            error_message = f"Agent '{agent_name}' execution returned None unexpectedly (potentially after timeout/cancel)."
            logger.error(error_message)
            return {"status": "failure", "result": None, "message": error_message}

        if not isinstance(structured_result, dict) or "status" not in structured_result:
            logger.warning(
                f"Agent '{agent_name}' returned an unexpected format: {structured_result}"
            )
            return {
                "status": "unknown",
                "result": structured_result,
                "message": f"Agent response format might be outdated or missing 'status': {str(structured_result)[:500]}",
            }

        # --- File Creation Verification (for agent_smith and tool_maker) ---
        if (
            agent_name in ["agent_smith", "tool_maker"]
            and structured_result.get("status") == "success"
        ):
            created_name = structured_result.get("result")
            if created_name:
                expected_path = ""
                if agent_name == "agent_smith":
                    expected_path = os.path.join("agents", f"{created_name}.py")
                elif agent_name == "tool_maker":
                    expected_path = os.path.join("tools", f"{created_name}.py")

                if expected_path and not os.path.exists(expected_path):
                    error_message = f"Agent '{agent_name}' reported success creating '{created_name}', but the expected file '{expected_path}' was not found on disk."
                    logger.error(error_message)
                    # Override the success status
                    structured_result["status"] = "failure"
                    structured_result["message"] = error_message
                    # structured_result['result'] = None # Keep result for context? Or clear it? Let's clear it.
                    structured_result["result"] = None
                elif expected_path:
                    logger.info(
                        f"Verified file exists for '{created_name}': {expected_path}"
                    )
                else:
                    # This case shouldn't happen if agent_name is one of the two, but handle defensively
                    logger.warning(
                        f"Could not determine expected path for agent '{agent_name}' result '{created_name}'. Skipping file verification."
                    )

            else:
                # Agent reported success but didn't return the name of the created entity
                logger.warning(
                    f"Agent '{agent_name}' reported success but did not return the name of the created entity in the 'result' field. Cannot verify file creation."
                )
                # Optionally, could change status to 'warning' or keep 'success' but add to message.
                # Let's add a note to the message but keep status as success for now.
                structured_result["message"] = (
                    structured_result.get("message", "")
                    + " (Warning: Could not verify file creation as agent result was missing)."
                )

        # --- Final Return ---
        print(
            f"Agent '{agent_name}' responded with status: {structured_result.get('status')}"
        )
        if structured_result.get("message"):
            print(
                f"Message: {str(structured_result['message'])[:500]}{'...' if len(str(structured_result['message'])) > 500 else ''}"
            )

        return structured_result

    except Exception as e:
        # Catch exceptions during agent loading or executor setup
        exception_trace = traceback.format_exc()
        error_message = f"An error occurred *before or during setup* for executing agent '{agent_name}' for task '{task[:100]}...':\n {e}\n{exception_trace}"
        logger.error(error_message)
        return {"status": "failure", "result": None, "message": error_message}
