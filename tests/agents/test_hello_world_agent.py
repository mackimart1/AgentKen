import unittest
from agents.hello_world_agent import hello_world_agent


class TestHelloWorldAgent(unittest.TestCase):
    def test_hello_world_agent_success(self):
        """Test that the hello_world_agent returns the expected greeting."""
        task = "Print a greeting"
        result = hello_world_agent(task)

        self.assertEqual(result["status"], "success")
        self.assertIn("Hello, world!", result["result"])
        self.assertEqual(result["message"], "Successfully printed greeting.")

    def test_hello_world_agent_error_handling(self):
        """Test that the hello_world_agent handles errors gracefully."""
        # Simulate an error (though this agent has no error-prone logic)
        result = hello_world_agent("")
        self.assertEqual(result["status"], "success")  # No error expected
