import unittest
import json
from tools import feature_engineer_for_agent_selection


class TestFeatureEngineerForAgentSelection(unittest.TestCase):

    def test_basic_feature_structuring(self):
        task_desc = "Analyze the recent stock market trends for tech companies."
        agent_meta = [
            {
                "name": "AnalystAgent",
                "capabilities": ["data_analysis", "reporting"],
                "type": "specialist",
            },
            {
                "name": "WebSearchAgent",
                "capabilities": ["web_search", "summarization"],
                "type": "generalist",
            },
        ]
        context = {"system_load": 0.75, "priority": "high"}

        expected_output = {
            "task_features": {
                "description": task_desc,
                "description_length": len(task_desc),
            },
            "agent_features": {
                "count": 2,
                "metadata_list": agent_meta,
            },
            "context_features": context,
        }

        # Use .invoke() for LangChain tools
        result = feature_engineer_for_agent_selection.feature_engineer_for_agent_selection.invoke(
            {
                "task_description": task_desc,
                "agent_metadata": agent_meta,
                "context_info": context,
            }
        )

        # Check if the structure and basic values match
        self.assertEqual(
            result["task_features"]["description"],
            expected_output["task_features"]["description"],
        )
        self.assertEqual(
            result["task_features"]["description_length"],
            expected_output["task_features"]["description_length"],
        )
        self.assertEqual(
            result["agent_features"]["count"],
            expected_output["agent_features"]["count"],
        )
        self.assertListEqual(
            result["agent_features"]["metadata_list"],
            expected_output["agent_features"]["metadata_list"],
        )
        self.assertDictEqual(
            result["context_features"], expected_output["context_features"]
        )

    def test_empty_inputs(self):
        task_desc = ""
        agent_meta = []
        context = {}

        expected_output = {
            "task_features": {
                "description": "",
                "description_length": 0,
            },
            "agent_features": {
                "count": 0,
                "metadata_list": [],
            },
            "context_features": {},
        }

        result = feature_engineer_for_agent_selection.feature_engineer_for_agent_selection.invoke(
            {
                "task_description": task_desc,
                "agent_metadata": agent_meta,
                "context_info": context,
            }
        )

        self.assertDictEqual(result, expected_output)

    def test_json_serializable_output(self):
        # Test if the output is JSON serializable (as checked in the tool)
        task_desc = "Test task"
        agent_meta = [{"name": "TestAgent"}]
        context = {"load": 0.1}

        result = feature_engineer_for_agent_selection.feature_engineer_for_agent_selection.invoke(
            {
                "task_description": task_desc,
                "agent_metadata": agent_meta,
                "context_info": context,
            }
        )

        try:
            json.dumps(result)
        except TypeError:
            self.fail("Output is not JSON serializable")


if __name__ == "__main__":
    unittest.main()
