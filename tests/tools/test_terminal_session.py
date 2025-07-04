import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Ensure the tools module is importable
try:
    from tools.terminal_session import terminal_session, _terminal_sessions
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.terminal_session import terminal_session, _terminal_sessions


class TestTerminalSession(unittest.TestCase):

    def setUp(self):
        """Clear sessions before each test."""
        _terminal_sessions.clear()

    def tearDown(self):
        """Clean up sessions after each test."""
        _terminal_sessions.clear()

    def test_create_session_success(self):
        """Test successful session creation."""
        result = terminal_session.invoke(
            {"action": "create", "session_id": "test_session"}
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("test_session", _terminal_sessions)
        self.assertIn("Terminal session 'test_session' created", result["message"])
        self.assertIn("session_id", result["data"])
        self.assertIn("working_directory", result["data"])

    def test_create_session_duplicate(self):
        """Test creating a session that already exists."""
        # Create first session
        terminal_session.invoke({"action": "create", "session_id": "duplicate"})

        # Try to create duplicate
        result = terminal_session.invoke(
            {"action": "create", "session_id": "duplicate"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIn("already exists", result["message"])

    def test_create_session_invalid_directory(self):
        """Test creating session with invalid working directory."""
        result = terminal_session.invoke(
            {
                "action": "create",
                "session_id": "test",
                "working_directory": "/nonexistent/path",
            }
        )

        self.assertEqual(result["status"], "failure")
        self.assertIn("does not exist", result["message"])

    def test_create_session_with_environment_vars(self):
        """Test creating session with environment variables."""
        env_vars = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}

        result = terminal_session.invoke(
            {"action": "create", "session_id": "env_test", "environment_vars": env_vars}
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["environment_vars"], env_vars)

    @patch("tools.terminal_session.subprocess.run")
    @patch("tools.terminal_session.time.time")
    def test_execute_command_success(self, mock_time, mock_subprocess_run):
        """Test successful command execution."""
        mock_time.side_effect = [
            0.0,
            1.0,
            1.5,
        ]  # session creation, command start, command end

        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = "Hello World"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Execute command (auto-creates session)
        result = terminal_session.invoke(
            {
                "action": "execute",
                "session_id": "auto_created",
                "command": "echo 'Hello World'",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["stdout"], "Hello World")
        self.assertEqual(result["data"]["returncode"], 0)
        self.assertIn("auto_created", _terminal_sessions)

    def test_execute_command_no_command(self):
        """Test executing without providing a command."""
        result = terminal_session.invoke({"action": "execute", "session_id": "test"})

        self.assertEqual(result["status"], "failure")
        self.assertIn("Command is required", result["message"])

    def test_execute_dangerous_command(self):
        """Test that dangerous commands are blocked."""
        dangerous_commands = ["rm -rf /", "del /f /s /q C:\\", "format c:"]

        for cmd in dangerous_commands:
            result = terminal_session.invoke(
                {"action": "execute", "session_id": "danger_test", "command": cmd}
            )

            self.assertEqual(result["status"], "failure")
            self.assertIn("blocked for security", result["message"])

    def test_execute_cd_command(self):
        """Test directory change command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create session
            terminal_session.invoke(
                {
                    "action": "create",
                    "session_id": "cd_test",
                    "working_directory": temp_dir,
                }
            )

            # Create a subdirectory
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)

            # Change to subdirectory
            result = terminal_session.invoke(
                {"action": "execute", "session_id": "cd_test", "command": f"cd subdir"}
            )

            self.assertEqual(result["status"], "success")
            self.assertIn("Changed directory", result["data"]["stdout"])

            # Verify working directory changed
            session = _terminal_sessions["cd_test"]
            self.assertEqual(session.working_directory, sub_dir)

    def test_execute_environment_variable_setting(self):
        """Test setting environment variables."""
        # Create session
        terminal_session.invoke({"action": "create", "session_id": "env_test"})

        # Set environment variable
        result = terminal_session.invoke(
            {
                "action": "execute",
                "session_id": "env_test",
                "command": "TEST_VAR=test_value",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("Set environment variable", result["data"]["stdout"])

        # Verify environment variable was set
        session = _terminal_sessions["env_test"]
        self.assertEqual(session.environment_vars["TEST_VAR"], "test_value")

    @patch("tools.terminal_session.subprocess.run")
    @patch("tools.terminal_session.time.time")
    def test_execute_multi_command(self, mock_time, mock_subprocess_run):
        """Test multi-command execution."""
        mock_time.side_effect = [
            0.0,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
        ]  # Multiple command timings

        # Mock subprocess results
        mock_results = [
            MagicMock(stdout="output1", stderr="", returncode=0),
            MagicMock(stdout="output2", stderr="", returncode=0),
        ]
        mock_subprocess_run.side_effect = mock_results

        multi_command = "echo 'first command'\necho 'second command'"

        result = terminal_session.invoke(
            {
                "action": "execute",
                "session_id": "multi_test",
                "command": multi_command,
                "multi_command": True,
            }
        )

        self.assertIn(result["status"], ["success", "partial"])
        self.assertEqual(result["data"]["commands_executed"], 2)
        self.assertEqual(result["data"]["successful"], 2)
        self.assertEqual(result["data"]["failed"], 0)

    def test_list_sessions(self):
        """Test listing all sessions."""
        # Create multiple sessions
        terminal_session.invoke({"action": "create", "session_id": "session1"})
        terminal_session.invoke({"action": "create", "session_id": "session2"})

        result = terminal_session.invoke({"action": "list"})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["session_count"], 2)
        self.assertEqual(len(result["data"]["sessions"]), 2)

        session_ids = [s["session_id"] for s in result["data"]["sessions"]]
        self.assertIn("session1", session_ids)
        self.assertIn("session2", session_ids)

    def test_destroy_session(self):
        """Test destroying a session."""
        # Create session
        terminal_session.invoke({"action": "create", "session_id": "to_destroy"})
        self.assertIn("to_destroy", _terminal_sessions)

        # Destroy session
        result = terminal_session.invoke(
            {"action": "destroy", "session_id": "to_destroy"}
        )

        self.assertEqual(result["status"], "success")
        self.assertNotIn("to_destroy", _terminal_sessions)
        self.assertIn("destroyed", result["message"])

    def test_destroy_nonexistent_session(self):
        """Test destroying a session that doesn't exist."""
        result = terminal_session.invoke(
            {"action": "destroy", "session_id": "nonexistent"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIn("not found", result["message"])

    def test_get_session_info(self):
        """Test getting session information."""
        # Create session with environment variables
        env_vars = {"TEST": "value"}
        terminal_session.invoke(
            {
                "action": "create",
                "session_id": "info_test",
                "environment_vars": env_vars,
            }
        )

        result = terminal_session.invoke({"action": "info", "session_id": "info_test"})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["session_id"], "info_test")
        self.assertEqual(result["data"]["environment_vars"], env_vars)
        self.assertIn("working_directory", result["data"])
        self.assertIn("created_at", result["data"])

    @patch("tools.terminal_session.subprocess.run")
    @patch("tools.terminal_session.time.time")
    def test_get_session_history(self, mock_time, mock_subprocess_run):
        """Test getting session command history."""
        mock_time.side_effect = [0.0, 1.0, 1.5]

        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Execute a command to create history
        terminal_session.invoke(
            {
                "action": "execute",
                "session_id": "history_test",
                "command": "echo 'test'",
            }
        )

        # Get history
        result = terminal_session.invoke(
            {"action": "history", "session_id": "history_test"}
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["total_commands"], 1)
        self.assertEqual(len(result["data"]["history"]), 1)
        self.assertEqual(result["data"]["history"][0]["command"], "echo 'test'")

    def test_invalid_action(self):
        """Test invalid action handling."""
        result = terminal_session.invoke({"action": "invalid_action"})

        self.assertEqual(result["status"], "failure")
        self.assertIn("Unknown action", result["message"])


if __name__ == "__main__":
    unittest.main()
