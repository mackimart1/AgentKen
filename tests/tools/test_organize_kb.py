import unittest
import sys
import os

# Add the parent directory (containing 'tools') to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tools import organize_kb  # Assuming tools package is importable


class TestOrganizeKb(unittest.TestCase):
    def test_smoke_test(self):
        """Smoke test to ensure the tool can be invoked."""
        try:
            result = organize_kb.organize_kb.invoke(
                {
                    "identifier": "doc123",
                    "organization_info": {"action": "add_tags", "tags": ["testing"]},
                }
            )
            self.assertIsInstance(result, str)  # Check if it returns a string
            self.assertIn("doc123", result)  # Basic check on output content
            self.assertIn("add_tags", result)
        except Exception as e:
            self.fail(f"organize_kb.invoke raised exception unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
