import unittest
import os
import sys
import warnings

# Ensure the tool module can be imported by adding the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Attempt to import the tool and check model status
_TOOL_IMPORTED = False
_MODEL_LOADED = False
ner_tool = None

try:
    # Suppress warnings during import test, especially download warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tools.ner_tool import (
            ner_tool as ner_tool_func,
            _model_loaded as ner_model_loaded_status,
        )
    ner_tool = ner_tool_func
    _TOOL_IMPORTED = True
    _MODEL_LOADED = ner_model_loaded_status
    if not _MODEL_LOADED:
        print(
            "Warning: spaCy model 'en_core_web_sm' was not loaded during import. Tests requiring the model will be skipped."
        )
except ImportError as e:
    # Check if it's specifically a spaCy import error
    if "spacy" in str(e).lower():
        print(f"ImportError: {e}. Skipping all ner_tool tests. Please install 'spacy'.")
    else:
        print(f"ImportError: {e}. Skipping all ner_tool tests.")

except Exception as e:  # Catch other potential issues during import/load
    print(f"An unexpected error occurred during import/setup: {e}. Skipping tests.")


@unittest.skipUnless(_TOOL_IMPORTED, "Skipping tests: ner_tool could not be imported.")
class TestNerTool(unittest.TestCase):

    @unittest.skipUnless(
        _MODEL_LOADED,
        "Skipping test: spaCy model 'en_core_web_sm' could not be loaded.",
    )
    def test_ner_extraction(self):
        """Test NER extraction with a standard sentence."""
        text = "Apple is looking at buying U.K. startup for $1 billion in London."
        expected_entities_options = [
            {"text": "Apple", "type": "ORG"},
            {"text": "U.K.", "type": "GPE"},
            {"text": "$1 billion", "type": "MONEY"},
            {"text": "London", "type": "GPE"},
        ]
        expected_set = set((d["text"], d["type"]) for d in expected_entities_options)

        result = ner_tool.invoke({"text": text})
        self.assertNotIn(
            "error", result, f"Tool returned an error: {result.get('error')}"
        )
        self.assertIn("entities", result)

        result_set = set((d["text"], d["type"]) for d in result["entities"])
        # Check if the result set contains all expected entities (more robust than exact match)
        self.assertTrue(
            expected_set.issubset(result_set),
            f"Expected {expected_set} to be a subset of {result_set}",
        )

    @unittest.skipUnless(
        _MODEL_LOADED,
        "Skipping test: spaCy model 'en_core_web_sm' could not be loaded.",
    )
    def test_no_entities(self):
        """Test NER with text containing no named entities."""
        text = "This is a simple sentence without proper nouns."
        expected = {"entities": []}
        result = ner_tool.invoke({"text": text})
        self.assertNotIn(
            "error", result, f"Tool returned an error: {result.get('error')}"
        )
        self.assertEqual(result, expected)

    # This test should run regardless of model load status, testing input validation
    def test_empty_input(self):
        """Test NER with empty string input."""
        text = ""
        expected = {"entities": []}
        result = ner_tool.invoke({"text": text})
        # If model isn't loaded, it should return an error
        if not _MODEL_LOADED:
            self.assertIn("error", result)
            self.assertIn("model", result.get("error", "").lower())
        else:
            # If model is loaded, it should return empty entities for empty input
            self.assertNotIn(
                "error", result, f"Tool returned an error: {result.get('error')}"
            )
            self.assertEqual(result, expected)

    # This test should run regardless of model load status, testing input validation
    def test_invalid_input_type(self):
        """Test NER with non-string input."""
        text = 12345
        expected = {"entities": []}  # Expect empty list for invalid input type
        result = ner_tool.invoke({"text": text})
        # If model isn't loaded, it should return an error
        if not _MODEL_LOADED:
            self.assertIn("error", result)
            self.assertIn("model", result.get("error", "").lower())
        else:
            # If model is loaded, it should handle non-string gracefully
            self.assertNotIn(
                "error", result, f"Tool returned an error: {result.get('error')}"
            )
            self.assertEqual(result, expected)

    @unittest.skipUnless(
        _MODEL_LOADED,
        "Skipping test: spaCy model 'en_core_web_sm' could not be loaded.",
    )
    def test_unicode_input(self):
        """Test NER with unicode characters."""
        text = "San José is a city in California, near München."
        expected_entities = [
            {"text": "San José", "type": "GPE"},
            {"text": "California", "type": "GPE"},
            {"text": "München", "type": "GPE"},  # spaCy often recognizes München as GPE
        ]
        expected_set = set((d["text"], d["type"]) for d in expected_entities)

        result = ner_tool.invoke({"text": text})
        self.assertNotIn(
            "error", result, f"Tool returned an error: {result.get('error')}"
        )
        self.assertIn("entities", result)

        result_set = set((d["text"], d["type"]) for d in result["entities"])
        self.assertTrue(
            expected_set.issubset(result_set),
            f"Expected {expected_set} to be a subset of {result_set}",
        )


if __name__ == "__main__":
    # Need to adjust argv for unittest when running script directly
    unittest.main(argv=[sys.argv[0]] + sys.argv[1:])
