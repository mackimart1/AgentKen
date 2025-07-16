import json
import logging
from typing import Dict, Any, Optional
from ..tools.secure_code_executor import secure_code_executor

# Set up logging
logger = logging.getLogger(__name__)


def code_executor(code: str, language: str = "python", timeout: int = 30) -> str:
    """
    Executes a code snippet in a secure environment and returns the result.

    Args:
        code (str): The code snippet to execute.
        language (str, optional): Programming language of the code. Defaults to 'python'.
        timeout (int, optional): Maximum execution time in seconds. Defaults to 30.

    Returns:
        str: A JSON string containing:
            - status: 'success' or 'failure'
            - stdout: Captured standard output (if any)
            - stderr: Captured standard error (if any)
            - error: Error information (if any)
            - message: Human-readable summary of execution
            - execution_time: Time taken for execution in seconds
    """
    logger.info(f"Code Executor Agent: Processing {language} code execution request")
    logger.debug(f"Code snippet to execute:\n{code}")

    output: Dict[str, Any] = {
        "status": "failure",
        "stdout": None,
        "stderr": None,
        "error": None,
        "message": "Code execution not implemented yet.",
        "execution_time": 0.0,
    }

    if not code or not code.strip():
        output["message"] = "Empty code snippet provided."
        return json.dumps(output, indent=2)

    try:
        # Sanitize and validate inputs
        validated_language = _validate_language(language)
        validated_timeout = _validate_timeout(timeout)

        # Call the secure code execution tool
        logger.info(
            f"Executing {validated_language} code with {validated_timeout}s timeout"
        )
        tool_result_json = secure_code_executor.invoke({
            "code": code,
            "language": validated_language
        })

        try:
            tool_result = json.loads(tool_result_json)
        except json.JSONDecodeError as json_err:
            raise ValueError(f"Received invalid JSON from executor tool: {json_err}")

        # Validate and populate the agent's output based on the tool's result
        output = _process_tool_result(tool_result)
        logger.info(f"Code execution completed with status: {output['status']}")

        if output["status"] == "success":
            logger.debug(f"Execution stdout: {output.get('stdout', '')[:100]}...")
        else:
            logger.warning(f"Execution failed: {output.get('error', 'Unknown error')}")

    except Exception as e:
        logger.exception("Error in code_executor agent")
        output["status"] = "failure"
        output["error"] = str(e)
        output["message"] = f"An error occurred within the code_executor agent: {e}"

    return json.dumps(output, indent=2)


def _validate_language(language: str) -> str:
    """
    Validates and normalizes the language parameter.

    Args:
        language (str): Language identifier

    Returns:
        str: Normalized language identifier

    Raises:
        ValueError: If language is invalid or unsupported
    """
    supported_languages = {"python", "javascript", "bash", "shell"}

    if not language or not isinstance(language, str):
        return "python"  # Default to Python

    language = language.lower().strip()

    # Handle common aliases
    if language in ("js", "node"):
        language = "javascript"
    elif language in ("sh", "zsh"):
        language = "bash"

    if language not in supported_languages:
        logger.warning(f"Unsupported language: {language}, defaulting to Python")
        return "python"

    return language


def _validate_timeout(timeout: int) -> int:
    """
    Validates and normalizes the timeout parameter.

    Args:
        timeout (int): Timeout value in seconds

    Returns:
        int: Normalized timeout value
    """
    if not isinstance(timeout, (int, float)):
        return 30  # Default timeout

    # Ensure timeout is within reasonable bounds
    min_timeout = 1
    max_timeout = 300  # 5 minutes max

    return max(min_timeout, min(int(timeout), max_timeout))


def _process_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes and validates the tool execution result.

    Args:
        result (Dict[str, Any]): Raw result from secure_code_executor

    Returns:
        Dict[str, Any]: Processed and validated output
    """
    output = {
        "status": "failure",
        "stdout": None,
        "stderr": None,
        "error": None,
        "message": "Execution completed with unknown status.",
        "execution_time": result.get("execution_time", 0.0),
    }

    # Set status
    if result.get("status") == "success":
        output["status"] = "success"
        output["message"] = "Code execution completed successfully."
    else:
        output["message"] = result.get("message", "Execution failed.")

    # Process stdout (ensure it's a string)
    if "stdout" in result:
        output["stdout"] = (
            str(result["stdout"]) if result["stdout"] is not None else None
        )

    # Process stderr (ensure it's a string)
    if "stderr" in result:
        output["stderr"] = (
            str(result["stderr"]) if result["stderr"] is not None else None
        )

    # Process error information
    if "error" in result:
        output["error"] = result["error"]
        if output["error"] and output["status"] == "success":
            # Error present but status was success - this is inconsistent
            output["status"] = "failure"
            output["message"] = f"Execution completed with errors: {output['error']}"

    return output


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Configure logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("===== Running Code Executor Agent Tests =====")

    # Test case 1: Basic functionality
    test_code = """
print('Hello from executed code!')
import sys
sys.stderr.write('This is an error message.\\n')
"""
    print("\n✅ Test Case 1: Basic functionality")
    result_json = code_executor(test_code)
    print(f"Result:\n{result_json}")

    # Test case 2: Error handling
    test_code_error = """
print('About to generate an error...')
print(1/0)  # Division by zero error
"""
    print("\n✅ Test Case 2: Error handling")
    result_json_error = code_executor(test_code_error)
    print(f"Result:\n{result_json_error}")

    # Test case 3: Empty code
    print("\n✅ Test Case 3: Empty code")
    result_json_empty = code_executor("")
    print(f"Result:\n{result_json_empty}")

    # Test case 4: Different language
    test_code_js = "console.log('Hello from JavaScript!');"
    print("\n✅ Test Case 4: JavaScript execution")
    result_json_js = code_executor(test_code_js, language="javascript")
    print(f"Result:\n{result_json_js}")
