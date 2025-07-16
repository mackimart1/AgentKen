import unittest
from agents.hello_world_agent import hello_world_agent


class TestHelloWorldAgent(unittest.TestCase):

    def test_hello_world_agent_success(self):
        """
        Tests that the hello_world_agent returns the expected greeting.
        """
        task = "Say hello"
        expected_result = {
            "status": "success",
            "result": "Hello, World!",
            "message": "Successfully generated a greeting.",
        }
        result = hello_world_agent(task)
        self.assertEqual(result, expected_result)

    def test_hello_world_agent_ignores_task(self):
        """
        Tests that the agent's output is the same regardless of the task input.
        """
        task1 = "Say hello"
        task2 = "Do something else"
        result1 = hello_world_agent(task1)
        result2 = hello_world_agent(task2)
        self.assertEqual(result1["result"], result2["result"])


if __name__ == "__main__":
    unittest.main()
