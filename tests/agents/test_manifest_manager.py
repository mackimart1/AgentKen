import unittest
import json
from agents.manifest_manager import ManifestManager


# Mock the default_api for testing file operations
class MockDefaultApi:
    def __init__(self):
        self.files = {}

    def read_file(self, file_path: str):
        if file_path in self.files:
            return {"content": self.files[file_path]}
        # Simulate FileNotFoundError for read_file
        raise FileNotFoundError

    def write_to_file(self, file: str, file_contents: str):
        self.files[file] = file_contents
        return {"status": "success", "message": f"File {file} written successfully."}

    def delete_file(self, file_path: str):
        if file_path in self.files:
            del self.files[file_path]
            return {
                "status": "success",
                "message": f"File {file_path} deleted successfully.",
            }
        return {"status": "error", "message": f"File {file_path} not found."}


class TestManifestManager(unittest.TestCase):

    def setUp(self):
        # Ensure a clean state before each test
        self.mock_api = MockDefaultApi()
        self.manifest_path = "agents_manifest.json"
        # Clear files in mock_api before each test
        self.mock_api.files = {}
        self.manager = ManifestManager(self.mock_api)

    def test_add_agent(self):
        result = self.manager.add_agent("test_agent", "A test agent", ["capability1"])
        self.assertEqual(result["status"], "success")
        self.assertIn(self.manifest_path, self.mock_api.files)

        manifest_content = json.loads(self.mock_api.files[self.manifest_path])
        self.assertEqual(len(manifest_content["agents"]), 1)
        self.assertEqual(manifest_content["agents"][0]["name"], "test_agent")

        # Test adding an existing agent
        result = self.manager.add_agent(
            "test_agent", "Another description", ["capability2"]
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("already exists", result["message"])

    def test_update_agent(self):
        self.manager.add_agent("test_agent", "A test agent", ["capability1"])
        result = self.manager.update_agent(
            "test_agent",
            new_description="Updated description",
            new_capabilities=["cap2"],
        )
        self.assertEqual(result["status"], "success")

        manifest_content = json.loads(self.mock_api.files[self.manifest_path])
        self.assertEqual(
            manifest_content["agents"][0]["description"], "Updated description"
        )
        self.assertEqual(manifest_content["agents"][0]["capabilities"], ["cap2"])

        # Test updating non-existent agent
        result = self.manager.update_agent("non_existent_agent", new_description="Desc")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_remove_agent(self):
        self.manager.add_agent("test_agent", "A test agent", ["capability1"])
        result = self.manager.remove_agent("test_agent")
        self.assertEqual(result["status"], "success")

        manifest_content = json.loads(self.mock_api.files[self.manifest_path])
        self.assertEqual(len(manifest_content["agents"]), 0)

        # Test removing non-existent agent
        result = self.manager.remove_agent("non_existent_agent")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_get_agent(self):
        self.manager.add_agent("test_agent", "A test agent", ["capability1"])
        result = self.manager.get_agent("test_agent")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["agent"]["name"], "test_agent")

        # Test getting non-existent agent
        result = self.manager.get_agent("non_existent_agent")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_list_agents(self):
        self.manager.add_agent("agent1", "Desc1", ["cap1"])
        self.manager.add_agent("agent2", "Desc2", ["cap2"])
        result = self.manager.list_agents()
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["agents"]), 2)
        self.assertEqual(result["agents"][0]["name"], "agent1")
        self.assertEqual(result["agents"][1]["name"], "agent2")

    def test_read_empty_manifest(self):
        # Ensure manifest file does not exist initially
        # The mock_api.files is cleared in setUp, so this is implicitly handled
        manifest = self.manager._read_manifest()
        self.assertEqual(manifest, {"agents": []})

    def test_malformed_json(self):
        self.mock_api.files[self.manifest_path] = "{invalid json"
        manifest = self.manager._read_manifest()
        self.assertEqual(manifest, {"agents": []})

    def test_write_manifest_failure(self):
        # Simulate write failure by making write_to_file return an error
        original_write_to_file = self.mock_api.write_to_file

        def mock_write_fail(*args, **kwargs):
            return {"status": "error", "message": "Disk full"}

        self.mock_api.write_to_file = mock_write_fail

        result = self.manager.add_agent("fail_agent", "Should fail", [])
        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to add agent", result["message"])

        self.mock_api.write_to_file = original_write_to_file  # Restore original


if __name__ == "__main__":
    unittest.main()
