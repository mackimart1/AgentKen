import unittest
import json

# Assuming the test runner configures path correctly to import from tools directory
from tools import format_annotation


class TestFormatAnnotation(unittest.TestCase):

    def test_format_annotation_success(self):
        log = "User request: Show me the weather in London."
        ann = "Intent: get_weather, Location: London"
        expected_output = json.dumps({"log_entry": log, "annotation": ann})
        # Invoke the tool with a dictionary payload
        result = format_annotation.format_annotation.invoke(
            {"log_entry": log, "annotation": ann}
        )
        self.assertEqual(result, expected_output)
        # Verify it's valid JSON
        self.assertIsInstance(json.loads(result), dict)

    def test_format_annotation_empty_strings(self):
        log = ""
        ann = ""
        expected_output = json.dumps({"log_entry": log, "annotation": ann})
        result = format_annotation.format_annotation.invoke(
            {"log_entry": log, "annotation": ann}
        )
        self.assertEqual(result, expected_output)
        self.assertIsInstance(json.loads(result), dict)

    # LangChain's tool invocation should handle Pydantic validation for input types.
    # Testing for non-string inputs directly might bypass the decorator's validation.
    # Let's rely on the framework's validation for type errors during invoke.


if __name__ == "__main__":
    unittest.main()
