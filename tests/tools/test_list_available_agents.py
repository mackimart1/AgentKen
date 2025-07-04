import unittest
from unittest.mock import patch

# Adjust import path if necessary
try:
    from tools.list_available_agents import list_available_agents
except ImportError:
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.list_available_agents import list_available_agents


class TestListAvailableAgents(unittest.TestCase):

    # Patch the all_agents function within the tool's module
    @patch("tools.list_available_agents.utils.all_agents")
    def test_list_agents_success(self, mock_all_agents):
        """Tests that the tool correctly calls utils.all_agents and returns its result."""

        # Configure the mock utils.all_agents to return a sample dictionary
        mock_agent_dict = {
            "agent_smith": "Designs and implements new agents.",
            "web_researcher": "Researches the web.",
            "tool_maker": "Creates new tools for agents to use.",
            # Note: hermes is excluded by default in utils.all_agents
        }
        mock_all_agents.return_value = mock_agent_dict

        # Invoke the tool (it takes no arguments)
        result = list_available_agents.invoke({})

        # Assert utils.all_agents was called (with default exclude)
        mock_all_agents.assert_called_once()  # Can add with() if defaults change

        # Assert the tool returned the exact dictionary from the mock
        self.assertEqual(result, mock_agent_dict)

    # Patch the all_agents function within the tool's module
    @patch("tools.list_available_agents.utils.all_agents")
    def test_list_agents_empty(self, mock_all_agents):
        """Tests the scenario where utils.all_agents returns an empty dictionary."""

        mock_all_agents.return_value = {}

        result = list_available_agents.invoke({})

        mock_all_agents.assert_called_once()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
