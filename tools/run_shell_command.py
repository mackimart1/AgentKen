import subprocess
import os
import shlex
import platform
from typing import Dict, Any, Optional
from langchain_core.tools import tool


@tool
def run_shell_command(
    command: str,
    timeout: int = 30,
    working_directory: Optional[str] = None,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """
    Run a shell command and return the output with enhanced error handling and security.

    Args:
        command (str): The shell command to execute
        timeout (int): Maximum execution time in seconds (default: 30)
        working_directory (str, optional): Directory to run the command in
        capture_output (bool): Whether to capture stdout/stderr (default: True)

    Returns:
        Dict[str, Any]: Dictionary containing:
            - status (str): 'success' or 'failure'
            - stdout (str): Standard output
            - stderr (str): Standard error
            - returncode (int): Process return code
            - message (str): Human-readable status message
            - execution_time (float): Time taken to execute

    Raises:
        ValueError: If command is empty or contains dangerous patterns
    """
    import time

    # Input validation
    if not command or not command.strip():
        return {
            "status": "failure",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "message": "Empty command provided",
            "execution_time": 0.0,
        }

    # Basic security checks - prevent obviously dangerous commands
    dangerous_patterns = [
        "rm -rf /",
        "del /f /s /q C:\\",
        "format c:",
        "mkfs",
        ":(){ :|:& };:",
        "sudo rm -rf",
        "dd if=/dev/zero",
    ]

    command_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in command_lower:
            return {
                "status": "failure",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "message": f"Command blocked for security: contains dangerous pattern '{pattern}'",
                "execution_time": 0.0,
            }

    print(f"Running shell command: {command}")
    if working_directory:
        print(f"Working directory: {working_directory}")

    start_time = time.time()

    try:
        # Validate working directory
        if working_directory and not os.path.exists(working_directory):
            return {
                "status": "failure",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "message": f"Working directory does not exist: {working_directory}",
                "execution_time": time.time() - start_time,
            }

        # Execute command with timeout
        # Use utf-8 encoding with error handling for better compatibility
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=working_directory,
            env=os.environ.copy(),  # Inherit environment variables
            encoding="utf-8",
            errors="replace",  # Replace problematic characters instead of failing
        )

        execution_time = time.time() - start_time

        # Determine status based on return code
        status = "success" if result.returncode == 0 else "failure"
        message = f"Command executed {'successfully' if status == 'success' else 'with errors'} in {execution_time:.2f}s"

        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "message": message,
            "execution_time": execution_time,
        }

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "status": "failure",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "message": f"Command timed out after {timeout} seconds",
            "execution_time": execution_time,
        }

    except subprocess.SubprocessError as e:
        execution_time = time.time() - start_time
        return {
            "status": "failure",
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "message": f"Subprocess error: {str(e)}",
            "execution_time": execution_time,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "status": "failure",
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "message": f"Unexpected error: {str(e)}",
            "execution_time": execution_time,
        }
