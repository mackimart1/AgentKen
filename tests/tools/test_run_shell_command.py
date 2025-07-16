import unittest
import os
import sys
import subprocess
import time
from unittest.mock import patch, MagicMock

# Ensure the tools module is importable
try:
    from tools.run_shell_command import run_shell_command
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.run_shell_command import run_shell_command


class TestRunShellCommand(unittest.TestCase):

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    def test_run_command_success(self, mock_time, mock_subprocess_run):
        """Tests successful command execution."""
        # Mock time for execution time calculation
        mock_time.side_effect = [0.0, 1.5]  # start_time, end_time

        # Configure the mock subprocess.run to simulate success
        mock_result = MagicMock()
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        command = "echo 'Success'"
        result = run_shell_command.invoke({"command": command})

        # Assert subprocess.run was called correctly
        mock_subprocess_run.assert_called_once_with(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=None,
            env=unittest.mock.ANY,
        )

        # Assert the result structure
        expected_keys = [
            "status",
            "stdout",
            "stderr",
            "returncode",
            "message",
            "execution_time",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stdout"], "Success output")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["execution_time"], 1.5)

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    def test_run_command_failure_return_code(self, mock_time, mock_subprocess_run):
        """Tests command execution failure (non-zero return code)."""
        mock_time.side_effect = [0.0, 0.8]

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error: Something went wrong."
        mock_result.returncode = 1
        mock_subprocess_run.return_value = mock_result

        command = "exit 1"
        result = run_shell_command.invoke({"command": command})

        mock_subprocess_run.assert_called_once()

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "Error: Something went wrong.")
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["execution_time"], 0.8)

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    def test_run_command_timeout(self, mock_time, mock_subprocess_run):
        """Tests command timeout handling."""
        mock_time.side_effect = [0.0, 30.0]
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("test_cmd", 30)

        command = "sleep 60"
        result = run_shell_command.invoke({"command": command, "timeout": 30})

        self.assertEqual(result["status"], "failure")
        self.assertIn("timed out", result["message"])
        self.assertEqual(result["returncode"], -1)
        self.assertEqual(result["execution_time"], 30.0)

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    def test_run_command_subprocess_error(self, mock_time, mock_subprocess_run):
        """Tests handling when subprocess.run raises a SubprocessError."""
        mock_time.side_effect = [0.0, 0.1]
        mock_subprocess_run.side_effect = subprocess.SubprocessError(
            "Failed to start process"
        )

        command = "invalid_command"
        result = run_shell_command.invoke({"command": command})

        self.assertEqual(result["status"], "failure")
        self.assertIn("Subprocess error", result["message"])
        self.assertEqual(result["returncode"], -1)
        self.assertEqual(result["execution_time"], 0.1)

    def test_empty_command(self):
        """Tests handling of empty command."""
        result = run_shell_command.invoke({"command": ""})

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["message"], "Empty command provided")
        self.assertEqual(result["returncode"], -1)

    def test_dangerous_command_blocked(self):
        """Tests that dangerous commands are blocked."""
        dangerous_commands = [
            "rm -rf /",
            "format c:",
            "sudo rm -rf /important",
            ":(){ :|:& };:",
        ]

        for cmd in dangerous_commands:
            result = run_shell_command.invoke({"command": cmd})
            self.assertEqual(result["status"], "failure")
            self.assertIn("blocked for security", result["message"])
            self.assertEqual(result["returncode"], -1)

    @patch("tools.run_shell_command.os.path.exists")
    def test_invalid_working_directory(self, mock_exists):
        """Tests handling of invalid working directory."""
        mock_exists.return_value = False

        result = run_shell_command.invoke(
            {"command": "echo test", "working_directory": "/nonexistent/path"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIn("Working directory does not exist", result["message"])

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    @patch("tools.run_shell_command.os.path.exists")
    def test_custom_working_directory(
        self, mock_exists, mock_time, mock_subprocess_run
    ):
        """Tests command execution with custom working directory."""
        mock_exists.return_value = True
        mock_time.side_effect = [0.0, 1.0]

        mock_result = MagicMock()
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        result = run_shell_command.invoke(
            {"command": "pwd", "working_directory": "/tmp"}
        )

        mock_subprocess_run.assert_called_once_with(
            "pwd",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/tmp",
            env=unittest.mock.ANY,
        )

        self.assertEqual(result["status"], "success")

    @patch("tools.run_shell_command.subprocess.run")
    @patch("time.time")
    def test_custom_timeout(self, mock_time, mock_subprocess_run):
        """Tests command execution with custom timeout."""
        mock_time.side_effect = [0.0, 5.0]

        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        result = run_shell_command.invoke({"command": "echo test", "timeout": 60})

        mock_subprocess_run.assert_called_once_with(
            "echo test",
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=None,
            env=unittest.mock.ANY,
        )


if __name__ == "__main__":
    unittest.main()
