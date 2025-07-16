import unittest
from unittest.mock import patch, MagicMock, ANY

# Adjust import path if necessary
try:
    from tools.assign_agent_to_task import assign_agent_to_task
    import utils  # Assuming utils is importable
except ImportError:
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.assign_agent_to_task import assign_agent_to_task
    import utils


class TestAssignAgentToTask(unittest.TestCase):

    @patch("tools.assign_agent_to_task.utils.load_module")
    def test_assign_agent_success(self, mock_load_module):
        """Tests successful agent assignment and structured response handling."""
        # Mock the agent function that load_module would return
        mock_agent_function = MagicMock()
        mock_agent_function.return_value = {
            "status": "success",
            "result": "Agent completed task.",
            "message": "Agent executed successfully.",
        }

        # Mock the module returned by load_module
        mock_module = MagicMock()
        # Make getattr(mock_module, 'some_agent') return our mock agent function
        setattr(mock_module, "some_agent", mock_agent_function)
        mock_load_module.return_value = mock_module

        agent_name = "some_agent"
        task = "Perform a test task"

        # Invoke the tool directly (it's decorated with @tool)
        result = assign_agent_to_task.invoke({"agent_name": agent_name, "task": task})

        # Assertions
        mock_load_module.assert_called_once_with(f"agents/{agent_name}.py")
        mock_agent_function.assert_called_once_with(task=task)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Agent completed task.")
        self.assertEqual(result["message"], "Agent executed successfully.")

    @patch("tools.assign_agent_to_task.utils.load_module")
    def test_assign_agent_failure_status(self, mock_load_module):
        """Tests handling when the agent returns a 'failure' status."""
        mock_agent_function = MagicMock()
        mock_agent_function.return_value = {
            "status": "failure",
            "result": None,
            "message": "Agent encountered an internal error.",
        }
        mock_module = MagicMock()
        setattr(mock_module, "failing_agent", mock_agent_function)
        mock_load_module.return_value = mock_module

        result = assign_agent_to_task.invoke(
            {"agent_name": "failing_agent", "task": "Task"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["result"])
        self.assertEqual(result["message"], "Agent encountered an internal error.")

    @patch("tools.assign_agent_to_task.utils.load_module")
    def test_assign_agent_execution_exception(self, mock_load_module):
        """Tests handling when the agent function itself raises an exception."""
        mock_agent_function = MagicMock()
        mock_agent_function.side_effect = Exception("Agent runtime error!")

        mock_module = MagicMock()
        setattr(mock_module, "error_agent", mock_agent_function)
        mock_load_module.return_value = mock_module

        result = assign_agent_to_task.invoke(
            {"agent_name": "error_agent", "task": "Task"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["result"])
        self.assertIn(
            "An error occurred while executing agent error_agent", result["message"]
        )
        self.assertIn(
            "Agent runtime error!", result["message"]
        )  # Check if original exception is in message

    @patch("tools.assign_agent_to_task.utils.load_module")
    def test_assign_agent_load_module_fails(self, mock_load_module):
        """Tests handling when loading the agent module fails."""
        mock_load_module.side_effect = ImportError("Cannot find module")

        result = assign_agent_to_task.invoke(
            {"agent_name": "nonexistent_agent", "task": "Task"}
        )

        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["result"])
        self.assertIn(
            "An error occurred while executing agent nonexistent_agent",
            result["message"],
        )
        self.assertIn("Cannot find module", result["message"])


if __name__ == "__main__":
    unittest.main()
