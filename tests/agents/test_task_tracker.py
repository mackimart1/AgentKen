
import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

# The agent to be tested
from agents.task_tracker import task_tracker

class TestTaskTrackerAgent(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.now = datetime.utcnow()
        self.overdue_task_id = "task_overdue_123"
        self.ontime_task_id = "task_ontime_456"

        self.mock_tasks = [
            {
                "id": self.overdue_task_id,
                "description": "A very important task that is late.",
                "status": "IN_PROGRESS",
                "created_at": (self.now - timedelta(minutes=120)).isoformat() + 'Z',
                "estimated_duration": 60, # minutes
            },
            {
                "id": self.ontime_task_id,
                "description": "A task that is still on schedule.",
                "status": "IN_PROGRESS",
                "created_at": (self.now - timedelta(minutes=30)).isoformat() + 'Z',
                "estimated_duration": 60, # minutes
            },
            {
                "id": "task_no_duration_789",
                "description": "A task without an estimated duration.",
                "status": "IN_PROGRESS",
                "created_at": (self.now - timedelta(days=5)).isoformat() + 'Z',
                "estimated_duration": None,
            },
            {
                "id": "task_bad_date_012",
                "description": "A task with a bad date format.",
                "status": "IN_PROGRESS",
                "created_at": "not-a-real-date",
                "estimated_duration": 30,
            }
        ]

    @patch('agents.task_tracker.assign_agent_to_task')
    @patch('agents.task_tracker.list_tasks')
    def test_one_overdue_task(self, mock_list_tasks, mock_assign_agent):
        """Test that one overdue task is correctly identified and reported."""
        # Arrange: list_tasks returns our mock list, assign_agent_to_task succeeds
        mock_list_tasks.return_value = json.dumps({
            "status": "success",
            "tasks": self.mock_tasks
        })
        mock_assign_agent.return_value = json.dumps({
            "status": "success",
            "message": "Task assigned."
        })

        # Act
        result = task_tracker("Check for overdue tasks")

        # Assert
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['result']['overdue_count'], 1)
        self.assertIn(self.overdue_task_id, result['result']['alerts_sent_for'])
        
        # Verify list_tasks was called correctly
        mock_list_tasks.assert_called_once_with(status_filter='IN_PROGRESS')
        
        # Verify assign_agent_to_task was called once with the correct details
        mock_assign_agent.assert_called_once()
        args, kwargs = mock_assign_agent.call_args
        self.assertEqual(kwargs['agent_name'], 'hermes')
        self.assertIn(self.overdue_task_id, kwargs['task'])
        self.assertIn("exceeded its estimated completion time", kwargs['task'])

    @patch('agents.task_tracker.assign_agent_to_task')
    @patch('agents.task_tracker.list_tasks')
    def test_no_overdue_tasks(self, mock_list_tasks, mock_assign_agent):
        """Test behavior when no tasks are overdue."""
        # Arrange: list_tasks returns only the on-time task
        ontime_tasks = [t for t in self.mock_tasks if t['id'] == self.ontime_task_id]
        mock_list_tasks.return_value = json.dumps({
            "status": "success",
            "tasks": ontime_tasks
        })

        # Act
        result = task_tracker("Check for overdue tasks")

        # Assert
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['result'], [])
        self.assertEqual(result['message'], "No overdue tasks found.")
        
        # Verify assign_agent_to_task was NOT called
        mock_assign_agent.assert_not_called()

    @patch('agents.task_tracker.list_tasks')
    def test_list_tasks_failure(self, mock_list_tasks):
        """Test that the agent handles a failure from the list_tasks tool."""
        # Arrange: list_tasks returns a failure
        mock_list_tasks.return_value = json.dumps({
            "status": "failure",
            "message": "Database connection error"
        })

        # Act
        result = task_tracker("Check for overdue tasks")

        # Assert
        self.assertEqual(result['status'], 'failure')
        self.assertIsNone(result['result'])
        self.assertIn("Failed to list tasks", result['message'])
        self.assertIn("Database connection error", result['message'])

    @patch('agents.task_tracker.assign_agent_to_task')
    @patch('agents.task_tracker.list_tasks')
    def test_assign_agent_failure(self, mock_list_tasks, mock_assign_agent):
        """Test how the agent handles a failure when assigning a task to hermes."""
        # Arrange: list_tasks returns an overdue task, but assign_agent_to_task fails
        overdue_tasks = [t for t in self.mock_tasks if t['id'] == self.overdue_task_id]
        mock_list_tasks.return_value = json.dumps({
            "status": "success",
            "tasks": overdue_tasks
        })
        mock_assign_agent.return_value = json.dumps({
            "status": "failure",
            "message": "Hermes agent is offline"
        })

        # Act
        result = task_tracker("Check for overdue tasks")

        # Assert
        self.assertEqual(result['status'], 'success') # The agent itself succeeds
        self.assertEqual(result['result']['overdue_count'], 1)
        self.assertEqual(result['result']['alerts_sent_for'], []) # Alert was not sent

    @patch('agents.task_tracker.list_tasks')
    def test_unexpected_exception(self, mock_list_tasks):
        """Test the main exception handler of the agent."""
        # Arrange: list_tasks raises an unexpected error
        mock_list_tasks.side_effect = ValueError("Something broke badly")

        # Act
        result = task_tracker("Check for overdue tasks")

        # Assert
        self.assertEqual(result['status'], 'failure')
        self.assertIsNone(result['result'])
        self.assertIn("An unexpected error occurred", result['message'])
        self.assertIn("Something broke badly", result['message'])

if __name__ == '__main__':
    unittest.main()
