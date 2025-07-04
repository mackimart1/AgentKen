import subprocess
import json
import sys
import tempfile
import os
from langchain_core.tools import tool


@tool
def secure_code_executor(code: str, language: str = "python") -> str:
    """
    Executes a code snippet using subprocess and returns the result.
    Note: This provides basic execution but is NOT a secure sandbox.
    For true security, consider containerization (e.g., Docker).

    Args:
        code: The code snippet to execute (string).
        language: The language of the code (currently only 'python' is supported).

    Returns:
        A JSON string containing the execution status, stdout, stderr,
        any errors, and a message.
    """
    output = {
        "status": "failure",
        "stdout": None,
        "stderr": None,
        "error": None,
        "message": "",
    }

    if language.lower() != "python":
        output["message"] = (
            f"Unsupported language: {language}. Only Python is currently supported."
        )
        return json.dumps(output, indent=2)

    # Create a temporary file to store the code
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)

        # Execute the temporary file using the same Python interpreter
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30,  # Add a timeout for safety
        )

        output["stdout"] = process.stdout
        output["stderr"] = process.stderr

        if process.returncode == 0:
            output["status"] = "success"
            output["message"] = "Code executed successfully."
        else:
            output["status"] = "failure"
            output["message"] = (
                f"Code execution failed with return code {process.returncode}."
            )
            # Include stderr in the error field for easier debugging if execution failed
            output["error"] = (
                process.stderr
                if process.stderr
                else "Execution failed with non-zero exit code."
            )

    except subprocess.TimeoutExpired:
        output["status"] = "failure"
        output["error"] = "TimeoutExpired"
        output["message"] = "Code execution timed out after 30 seconds."
    except Exception as e:
        output["status"] = "failure"
        output["error"] = str(e)
        output["message"] = (
            f"An error occurred during code execution setup or processing: {e}"
        )
    finally:
        # Clean up the temporary file
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return json.dumps(output, indent=2)


# Example usage (for testing purposes)
if __name__ == "__main__":
    test_code_success = "print('Hello from secure executor!')\nimport sys\nsys.stderr.write('This is a stderr message.\\n')"
    result_json_success = secure_code_executor(test_code_success)
    print("--- Success Test ---")
    print(result_json_success)

    test_code_error = "print(1/0)"
    result_json_error = secure_code_executor(test_code_error)
    print("\n--- Error Test ---")
    print(result_json_error)

    test_code_timeout = "import time\ntime.sleep(40)\nprint('Should not see this')"
    result_json_timeout = secure_code_executor(test_code_timeout)
    print("\n--- Timeout Test ---")
    print(result_json_timeout)
