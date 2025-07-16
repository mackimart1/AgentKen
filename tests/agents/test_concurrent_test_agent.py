import unittest
from agents.concurrent_test_agent import concurrent_test_agent


class TestConcurrentTestAgent(unittest.TestCase):
    def test_concurrent_execution(self):
        """Test basic concurrent execution of tasks."""
        tasks = ["task1", "task2", "task3"]
        result = concurrent_test_agent(tasks)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["result"]), len(tasks))
        for task in tasks:
            self.assertIn(task, result["result"])
            self.assertTrue("completed successfully" in result["result"][task])

    def test_empty_task_list(self):
        """Test handling of an empty task list."""
        tasks = []
        result = concurrent_test_agent(tasks)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["result"]), 0)
        self.assertEqual(
            result["message"], "Successfully executed 0 tasks concurrently"
        )

    def test_error_handling(self):
        """Test error handling during concurrent execution."""
        # Simulate a task that raises an exception
        tasks = ["task1", "task2", "task3"]
        # Mock the execute_task function to raise an exception for "task2"
        original_execute_task = concurrent_test_agent.__globals__["execute_task"]

        def mock_execute_task(t: str) -> str:
            if t == "task2":
                raise ValueError("Simulated error")
            return original_execute_task(t)

        concurrent_test_agent.__globals__["execute_task"] = mock_execute_task

        result = concurrent_test_agent(tasks)

        # Restore the original function
        concurrent_test_agent.__globals__["execute_task"] = original_execute_task

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["result"]), len(tasks))
        self.assertIn("failed: Simulated error", result["result"]["task2"])


if __name__ == "__main__":
    unittest.main()
