import unittest
from unittest.mock import MagicMock, patch

# Adjust the import path as necessary for your project structure
# Assuming agents are in a directory named 'agents' at the root
from agents.memory_manager import (
    create_memory_manager_agent,
    delete_memory,
    list_memory_keys,
    retrieve_memory,
    store_memory,
)


class TestMemoryManagerAgent(unittest.TestCase):

    @patch("agents.memory_manager.scratchpad")
    def test_store_memory_success(self, mock_scratchpad):
        mock_scratchpad.return_value = {"status": "success", "key": "test_key"}
        result = store_memory("test_key", "test_value")
        self.assertIn("Successfully stored memory", result)
        mock_scratchpad.assert_called_once_with(
            action="write", key="test_key", value="test_value"
        )

    @patch("agents.memory_manager.scratchpad")
    def test_store_memory_failure(self, mock_scratchpad):
        mock_scratchpad.return_value = {"status": "failure"}
        result = store_memory("test_key", "test_value")
        self.assertIn("Failed to store memory", result)
        mock_scratchpad.assert_called_once_with(
            action="write", key="test_key", value="test_value"
        )

    @patch("agents.memory_manager.scratchpad")
    def test_retrieve_memory_success(self, mock_scratchpad):
        mock_scratchpad.return_value = "retrieved_value"
        result = retrieve_memory("test_key")
        self.assertIn("Retrieved memory for key 'test_key': retrieved_value", result)
        mock_scratchpad.assert_called_once_with(action="read", key="test_key")

    @patch("agents.memory_manager.scratchpad")
    def test_retrieve_memory_not_found(self, mock_scratchpad):
        mock_scratchpad.return_value = None
        result = retrieve_memory("non_existent_key")
        self.assertIn("No memory found for key: non_existent_key", result)
        mock_scratchpad.assert_called_once_with(action="read", key="non_existent_key")

    @patch("agents.memory_manager.scratchpad")
    def test_delete_memory_success(self, mock_scratchpad):
        mock_scratchpad.return_value = {"status": "success", "key": "test_key"}
        result = delete_memory("test_key")
        self.assertIn("Successfully deleted memory", result)
        mock_scratchpad.assert_called_once_with(action="delete", key="test_key")

    @patch("agents.memory_manager.scratchpad")
    def test_delete_memory_not_found(self, mock_scratchpad):
        mock_scratchpad.return_value = {
            "status": "key_not_found",
            "key": "non_existent_key",
        }
        result = delete_memory("non_existent_key")
        self.assertIn("No memory found to delete", result)
        mock_scratchpad.assert_called_once_with(action="delete", key="non_existent_key")

    @patch("agents.memory_manager.scratchpad")
    def test_list_memory_keys_success(self, mock_scratchpad):
        mock_scratchpad.return_value = ["key1", "key2"]
        result = list_memory_keys()
        self.assertIn("Currently stored memory keys: key1, key2", result)
        mock_scratchpad.assert_called_once_with(action="list")

    @patch("agents.memory_manager.scratchpad")
    def test_list_memory_keys_empty(self, mock_scratchpad):
        mock_scratchpad.return_value = []
        result = list_memory_keys()
        self.assertIn("No memory keys currently stored.", result)
        mock_scratchpad.assert_called_once_with(action="list")

    @patch("agents.memory_manager.scratchpad")
    def test_list_memory_keys_failure(self, mock_scratchpad):
        mock_scratchpad.return_value = {"error": "some error"}
        result = list_memory_keys()
        self.assertIn("Failed to list memory keys", result)
        mock_scratchpad.assert_called_once_with(action="list")

    # Test the agent creation and basic graph structure (without full execution)
    def test_create_memory_manager_agent(self):
        mock_llm = MagicMock()
        agent = create_memory_manager_agent(mock_llm)
        self.assertIsNotNone(agent)
        # You can add more assertions here to check the graph structure
        # For example, checking if nodes and edges are as expected.
        # This might require inspecting the internal structure of the compiled graph,
        # which can be complex. For now, just checking if it compiles is a good start.


if __name__ == "__main__":
    unittest.main()
