import unittest
from unittest.mock import patch, MagicMock

# Adjust import path if necessary
try:
    from agents.tool_maker import tool_maker
except ImportError:
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from agents.tool_maker import tool_maker


class TestToolMaker(unittest.TestCase):

    @patch("agents.tool_maker.graph")  # Mock the internal LangGraph graph
    def test_tool_maker_invocation_structure(self, mock_graph_invoke):
        """
        Tests that calling tool_maker returns the expected dictionary structure on success.
        Mocks the internal graph execution.
        """
        # Configure the mock graph to return a simulated final state indicating success
        mock_final_state = {
            "messages": [
                MagicMock(content="Successfully created and tested tool: 'new_tool'")
            ]
        }
        mock_graph_invoke.invoke.return_value = mock_final_state

        task = "Create a simple test tool"
        result = tool_maker(task=task)

        # Assert that the function was called
        mock_graph_invoke.invoke.assert_called_once()

        # Assert the returned structure and basic content
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("result", result)
        self.assertIn("message", result)

        # Check the parsed values based on our logic in tool_maker
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "new_tool")
        self.assertEqual(
            result["message"], "Successfully created and tested tool: 'new_tool'"
        )

    @patch("agents.tool_maker.graph")  # Mock the internal LangGraph graph
    def test_tool_maker_failure_structure(self, mock_graph_invoke):
        """
        Tests that tool_maker returns a failure structure if the final message indicates failure.
        """
        # Configure the mock graph to return a simulated final state indicating failure
        mock_final_state = {
            "messages": [MagicMock(content="Failed to create tool, test failed.")]
        }
        mock_graph_invoke.invoke.return_value = mock_final_state

        task = "Create a faulty tool"
        result = tool_maker(task=task)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "failure")
        self.assertIsNone(result["result"])  # No tool name expected on failure
        self.assertEqual(result["message"], "Failed to create tool, test failed.")


if __name__ == "__main__":
    unittest.main()
