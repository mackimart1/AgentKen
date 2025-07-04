import unittest
from unittest.mock import patch, MagicMock

# Assuming agent_smith is importable. Adjust if necessary based on project structure.
# Might need to adjust sys.path or use relative imports depending on how tests are run.
try:
    from agents.agent_smith import agent_smith, workflow_manager
except ImportError:
    # Handle potential import issues if running script directly vs. via test runner
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from agents.agent_smith import agent_smith, workflow_manager


class TestAgentSmith(unittest.TestCase):

    @patch("agents.agent_smith.workflow_manager")  # Mock the workflow manager
    def test_agent_smith_invocation_structure(self, mock_workflow_manager):
        """
        Tests that calling agent_smith returns the expected dictionary structure.
        Mocks the internal workflow execution.
        """
        # Create a mock workflow with invoke method
        mock_workflow = MagicMock()
        mock_workflow_manager.workflow = mock_workflow

        # Configure the mock workflow to return a simulated final state
        success_message = (
            "Successfully created, formatted, linted, and tested agent: 'test_agent'"
        )
        mock_final_state = {"messages": [MagicMock(content=success_message)]}
        mock_workflow.invoke.return_value = mock_final_state

        # Mock the _extract_content method to return the expected message
        mock_workflow_manager._extract_content.return_value = success_message

        # Mock the creation state to simulate successful completion
        with patch("agents.agent_smith.AgentCreationState") as MockState:
            from agents.agent_smith import AgentCreationPhase

            mock_state_instance = MagicMock()
            mock_state_instance.is_complete = True
            mock_state_instance.current_phase = AgentCreationPhase.COMPLETE
            mock_state_instance.agent_name = "test_agent"
            mock_state_instance.files_written = [
                "agents/test_agent.py",
                "tests/agents/test_test_agent.py",
            ]
            mock_state_instance.errors = []
            MockState.return_value = mock_state_instance

            task = "Create a simple test agent"
            result = agent_smith(task=task)

            # Assert that the workflow was called
            mock_workflow.invoke.assert_called_once()

            # Assert the returned structure and basic content
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertIn("result", result)
            self.assertIn("message", result)
            self.assertIn("phase", result)
            self.assertIn("files_created", result)
            self.assertIn("errors", result)

            # Check the parsed values based on our logic in agent_smith
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["result"], "test_agent")
            self.assertIn("successfully created", result["message"].lower())

    @patch("agents.agent_smith.workflow_manager")  # Mock the workflow manager
    def test_agent_smith_failure_structure(self, mock_workflow_manager):
        """
        Tests that agent_smith returns a failure structure if the final message indicates failure.
        """
        # Create a mock workflow with invoke method
        mock_workflow = MagicMock()
        mock_workflow_manager.workflow = mock_workflow

        # Configure the mock workflow to return a simulated final state indicating failure
        failure_message = "Failed to create agent due to error."
        mock_final_state = {"messages": [MagicMock(content=failure_message)]}
        mock_workflow.invoke.return_value = mock_final_state

        # Mock the _extract_content method to return the expected message
        mock_workflow_manager._extract_content.return_value = failure_message

        # Mock the creation state to simulate failure
        with patch("agents.agent_smith.AgentCreationState") as MockState:
            from agents.agent_smith import AgentCreationPhase

            mock_state_instance = MagicMock()
            mock_state_instance.is_complete = False
            mock_state_instance.current_phase = AgentCreationPhase.FAILED
            mock_state_instance.agent_name = None
            mock_state_instance.files_written = []
            mock_state_instance.errors = ["Test error"]
            MockState.return_value = mock_state_instance

            task = "Create a faulty agent"
            result = agent_smith(task=task)

            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "failure")
            self.assertIsNone(result["result"])  # No agent name expected on failure
            self.assertEqual(result["message"], "Failed to create agent due to error.")
            self.assertEqual(result["phase"], "failed")


if __name__ == "__main__":
    unittest.main()
