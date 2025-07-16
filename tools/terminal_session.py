"""
Advanced Terminal Session Tool for AgentK

Provides persistent terminal sessions with state management, command history,
and advanced features like environment variable management and multi-command execution.
"""

import subprocess
import os
import tempfile
import json
import time
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from langchain_core.tools import tool


@dataclass
class TerminalSession:
    """Represents a persistent terminal session."""

    session_id: str
    working_directory: str = field(default_factory=os.getcwd)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    command_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used = time.time()

    def add_to_history(self, command: str, result: Dict[str, Any]):
        """Add a command and its result to the history."""
        self.command_history.append(
            {"timestamp": time.time(), "command": command, "result": result}
        )
        # Keep only last 100 commands
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]


# Global session storage
_terminal_sessions: Dict[str, TerminalSession] = {}


@tool
def terminal_session(
    action: str,
    session_id: str = "default",
    command: Optional[str] = None,
    working_directory: Optional[str] = None,
    environment_vars: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    multi_command: bool = False,
) -> Dict[str, Any]:
    """
    Advanced terminal session management with persistent state.

    Args:
        action (str): Action to perform - 'create', 'execute', 'list', 'destroy', 'info', 'history'
        session_id (str): Unique identifier for the terminal session
        command (str, optional): Command to execute (required for 'execute' action)
        working_directory (str, optional): Directory to change to
        environment_vars (dict, optional): Environment variables to set
        timeout (int): Command timeout in seconds
        multi_command (bool): Whether to treat command as multiple commands separated by newlines

    Returns:
        Dict[str, Any]: Result dictionary with status, data, and message
    """

    if action == "create":
        return _create_session(session_id, working_directory, environment_vars)
    elif action == "execute":
        return _execute_command(session_id, command, timeout, multi_command)
    elif action == "list":
        return _list_sessions()
    elif action == "destroy":
        return _destroy_session(session_id)
    elif action == "info":
        return _get_session_info(session_id)
    elif action == "history":
        return _get_session_history(session_id)
    else:
        return {
            "status": "failure",
            "data": None,
            "message": f"Unknown action: {action}. Available actions: create, execute, list, destroy, info, history",
        }


def _create_session(
    session_id: str,
    working_directory: Optional[str],
    environment_vars: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Create a new terminal session."""
    if session_id in _terminal_sessions:
        return {
            "status": "failure",
            "data": None,
            "message": f"Session '{session_id}' already exists",
        }

    # Validate working directory
    if working_directory:
        if not os.path.exists(working_directory):
            return {
                "status": "failure",
                "data": None,
                "message": f"Working directory does not exist: {working_directory}",
            }
        if not os.path.isdir(working_directory):
            return {
                "status": "failure",
                "data": None,
                "message": f"Path is not a directory: {working_directory}",
            }

    # Create session
    session = TerminalSession(
        session_id=session_id,
        working_directory=working_directory or os.getcwd(),
        environment_vars=environment_vars or {},
    )

    _terminal_sessions[session_id] = session

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "working_directory": session.working_directory,
            "environment_vars": session.environment_vars,
            "created_at": session.created_at,
        },
        "message": f"Terminal session '{session_id}' created successfully",
    }


def _execute_command(
    session_id: str, command: Optional[str], timeout: int, multi_command: bool
) -> Dict[str, Any]:
    """Execute a command in the specified session."""
    if not command:
        return {
            "status": "failure",
            "data": None,
            "message": "Command is required for execute action",
        }

    # Get or create session
    if session_id not in _terminal_sessions:
        # Auto-create default session
        create_result = _create_session(session_id, None, None)
        if create_result["status"] != "success":
            return create_result

    session = _terminal_sessions[session_id]
    session.update_last_used()

    # Security check
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
                "data": None,
                "message": f"Command blocked for security: contains dangerous pattern '{pattern}'",
            }

    # Handle multi-command execution
    if multi_command:
        commands = [cmd.strip() for cmd in command.split("\n") if cmd.strip()]
        return _execute_multiple_commands(session, commands, timeout)
    else:
        return _execute_single_command(session, command, timeout)


def _execute_single_command(
    session: TerminalSession, command: str, timeout: int
) -> Dict[str, Any]:
    """Execute a single command in the session."""
    start_time = time.time()

    try:
        # Prepare environment
        env = os.environ.copy()
        env.update(session.environment_vars)

        # Handle directory changes
        if command.strip().startswith("cd "):
            new_dir = command.strip()[3:].strip()
            if new_dir:
                # Handle relative paths
                if not os.path.isabs(new_dir):
                    new_dir = os.path.join(session.working_directory, new_dir)

                new_dir = os.path.abspath(new_dir)

                if os.path.exists(new_dir) and os.path.isdir(new_dir):
                    session.working_directory = new_dir
                    result = {
                        "status": "success",
                        "stdout": f"Changed directory to: {new_dir}",
                        "stderr": "",
                        "returncode": 0,
                        "execution_time": time.time() - start_time,
                        "working_directory": new_dir,
                    }
                else:
                    result = {
                        "status": "failure",
                        "stdout": "",
                        "stderr": f"Directory not found: {new_dir}",
                        "returncode": 1,
                        "execution_time": time.time() - start_time,
                        "working_directory": session.working_directory,
                    }
            else:
                # cd with no arguments - go to home directory
                home_dir = os.path.expanduser("~")
                session.working_directory = home_dir
                result = {
                    "status": "success",
                    "stdout": f"Changed directory to: {home_dir}",
                    "stderr": "",
                    "returncode": 0,
                    "execution_time": time.time() - start_time,
                    "working_directory": home_dir,
                }

        # Handle environment variable setting
        elif "=" in command and not command.strip().startswith(
            ("echo", "printf", "test", "[")
        ):
            # Simple environment variable assignment
            if " " not in command.strip():
                var_name, var_value = command.strip().split("=", 1)
                session.environment_vars[var_name] = var_value
                result = {
                    "status": "success",
                    "stdout": f"Set environment variable: {var_name}={var_value}",
                    "stderr": "",
                    "returncode": 0,
                    "execution_time": time.time() - start_time,
                    "working_directory": session.working_directory,
                }
            else:
                # Execute as regular command
                result = _run_subprocess_command(
                    command, session, env, timeout, start_time
                )
        else:
            # Regular command execution
            result = _run_subprocess_command(command, session, env, timeout, start_time)

        # Add to history
        session.add_to_history(command, result)

        return {
            "status": result["status"],
            "data": result,
            "message": f"Command executed in session '{session.session_id}'",
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_result = {
            "status": "failure",
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "execution_time": execution_time,
            "working_directory": session.working_directory,
        }

        session.add_to_history(command, error_result)

        return {
            "status": "failure",
            "data": error_result,
            "message": f"Error executing command: {str(e)}",
        }


def _run_subprocess_command(
    command: str,
    session: TerminalSession,
    env: Dict[str, str],
    timeout: int,
    start_time: float,
) -> Dict[str, Any]:
    """Run a command using subprocess."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=session.working_directory,
            env=env,
        )

        execution_time = time.time() - start_time
        status = "success" if result.returncode == 0 else "failure"

        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": execution_time,
            "working_directory": session.working_directory,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "failure",
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1,
            "execution_time": time.time() - start_time,
            "working_directory": session.working_directory,
        }


def _execute_multiple_commands(
    session: TerminalSession, commands: List[str], timeout: int
) -> Dict[str, Any]:
    """Execute multiple commands in sequence."""
    results = []
    total_start_time = time.time()

    for i, cmd in enumerate(commands):
        if not cmd.strip():
            continue

        print(f"Executing command {i+1}/{len(commands)}: {cmd}")

        cmd_result = _execute_single_command(session, cmd, timeout)
        results.append({"command": cmd, "result": cmd_result["data"]})

        # Stop on first failure if it's a critical command
        if cmd_result["status"] == "failure" and cmd_result["data"]["returncode"] != 0:
            # Continue for non-critical failures, but note them
            print(f"Warning: Command failed: {cmd}")

    total_execution_time = time.time() - total_start_time

    # Aggregate results
    successful_commands = sum(1 for r in results if r["result"]["status"] == "success")
    failed_commands = len(results) - successful_commands

    overall_status = (
        "success"
        if failed_commands == 0
        else "partial" if successful_commands > 0 else "failure"
    )

    return {
        "status": overall_status,
        "data": {
            "commands_executed": len(results),
            "successful": successful_commands,
            "failed": failed_commands,
            "total_execution_time": total_execution_time,
            "results": results,
            "working_directory": session.working_directory,
        },
        "message": f"Executed {len(results)} commands: {successful_commands} successful, {failed_commands} failed",
    }


def _list_sessions() -> Dict[str, Any]:
    """List all active terminal sessions."""
    sessions_info = []
    current_time = time.time()

    for session_id, session in _terminal_sessions.items():
        sessions_info.append(
            {
                "session_id": session_id,
                "working_directory": session.working_directory,
                "environment_vars_count": len(session.environment_vars),
                "command_history_count": len(session.command_history),
                "created_at": session.created_at,
                "last_used": session.last_used,
                "age_seconds": current_time - session.created_at,
                "idle_seconds": current_time - session.last_used,
            }
        )

    return {
        "status": "success",
        "data": {"session_count": len(sessions_info), "sessions": sessions_info},
        "message": f"Found {len(sessions_info)} active terminal sessions",
    }


def _destroy_session(session_id: str) -> Dict[str, Any]:
    """Destroy a terminal session."""
    if session_id not in _terminal_sessions:
        return {
            "status": "failure",
            "data": None,
            "message": f"Session '{session_id}' not found",
        }

    session = _terminal_sessions.pop(session_id)

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "commands_executed": len(session.command_history),
            "session_duration": time.time() - session.created_at,
        },
        "message": f"Terminal session '{session_id}' destroyed",
    }


def _get_session_info(session_id: str) -> Dict[str, Any]:
    """Get detailed information about a session."""
    if session_id not in _terminal_sessions:
        return {
            "status": "failure",
            "data": None,
            "message": f"Session '{session_id}' not found",
        }

    session = _terminal_sessions[session_id]
    current_time = time.time()

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "working_directory": session.working_directory,
            "environment_vars": session.environment_vars,
            "command_history_count": len(session.command_history),
            "created_at": session.created_at,
            "last_used": session.last_used,
            "age_seconds": current_time - session.created_at,
            "idle_seconds": current_time - session.last_used,
        },
        "message": f"Session '{session_id}' information retrieved",
    }


def _get_session_history(session_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get command history for a session."""
    if session_id not in _terminal_sessions:
        return {
            "status": "failure",
            "data": None,
            "message": f"Session '{session_id}' not found",
        }

    session = _terminal_sessions[session_id]
    history = session.command_history[-limit:] if limit > 0 else session.command_history

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "total_commands": len(session.command_history),
            "showing": len(history),
            "history": history,
        },
        "message": f"Retrieved {len(history)} commands from session '{session_id}' history",
    }
