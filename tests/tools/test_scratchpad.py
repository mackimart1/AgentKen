import unittest
import os
import sys

# Ensure the tools directory is in the Python path
# Navigate up two levels from tests/tools/ to the project root, then into tools/
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
tools_dir = os.path.join(base_dir, "tools")
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)
# Add base_dir too, in case other imports are needed relative to the project root
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import the tool definition
try:
    # The file is scratchpad.py, the tool function is also named scratchpad
    # We import the module first
    import scratchpad as scratchpad_module

    # Access the tool object (decorated function) within the module
    scratchpad_tool = scratchpad_module.scratchpad
except ImportError as e:
    print(f"Error importing scratchpad tool: {e}")
    print(f"Current sys.path: {sys.path}")
    # Raise a more informative error if import fails
    raise ImportError(
        f"Could not import scratchpad tool from {tools_dir}. Check PYTHONPATH and file location. sys.path: {sys.path}"
    ) from e

# IMPORTANT ASSUMPTION: The test execution environment is set up such that
# calling scratchpad_tool.invoke(...) correctly triggers the
# default_api.scratchpad function provided in the agent's context.
# These tests verify the interaction with that API based on its documented behavior.


class TestScratchpadTool(unittest.TestCase):

    def setUp(self):
        """Clear the scratchpad before each test."""
        try:
            # Use invoke on the LangChain tool object
            result = scratchpad_tool.invoke({"action": "clear_all"})
            # Check if the API call itself indicated a problem (optional but good practice)
            if isinstance(result, dict) and result.get("status") != "success":
                print(
                    f"Warning: setUp clear_all API call did not return success: {result}"
                )
        except Exception as e:
            # If the invoke call itself fails, it's a critical setup error
            self.fail(
                f"Critical Error: setUp clear_all failed during tool invocation: {e}"
            )

    def tearDown(self):
        """Clear the scratchpad after each test to leave a clean state."""
        try:
            scratchpad_tool.invoke({"action": "clear_all"})
        except Exception as e:
            # Log warning if cleanup fails, but don't fail the test itself
            print(f"Warning: tearDown clear_all failed during tool invocation: {e}")

    def test_write_and_read(self):
        """Test writing a value and reading it back."""
        write_result = scratchpad_tool.invoke(
            {"action": "write", "key": "test_key_1", "value": "test_value_1"}
        )
        self.assertEqual(write_result, {"status": "success", "key": "test_key_1"})

        read_result = scratchpad_tool.invoke({"action": "read", "key": "test_key_1"})
        self.assertEqual(read_result, "test_value_1")

    def test_read_nonexistent(self):
        """Test reading a key that does not exist."""
        read_result = scratchpad_tool.invoke(
            {"action": "read", "key": "nonexistent_key"}
        )
        # Based on API doc: "Returns the stored value (str) or None if the key is not found."
        self.assertIsNone(read_result)

    def test_list_keys(self):
        """Test listing keys when empty and after adding keys."""
        # List empty scratchpad
        keys = scratchpad_tool.invoke({"action": "list"})
        self.assertEqual(keys, [])

        # Write some keys
        scratchpad_tool.invoke({"action": "write", "key": "key_a", "value": "val_a"})
        scratchpad_tool.invoke({"action": "write", "key": "key_b", "value": "val_b"})

        # List again
        keys = scratchpad_tool.invoke({"action": "list"})
        # Use assertCountEqual for order-independent comparison of lists
        self.assertCountEqual(keys, ["key_a", "key_b"])

    def test_delete_key(self):
        """Test deleting an existing key and trying to delete a non-existent key."""
        # Write a key
        scratchpad_tool.invoke(
            {"action": "write", "key": "to_delete", "value": "delete_me"}
        )

        # Check it's there
        read_result = scratchpad_tool.invoke({"action": "read", "key": "to_delete"})
        self.assertEqual(read_result, "delete_me")

        # Delete the key
        delete_result = scratchpad_tool.invoke({"action": "delete", "key": "to_delete"})
        self.assertEqual(delete_result, {"status": "success", "key": "to_delete"})

        # Check it's gone
        read_result = scratchpad_tool.invoke({"action": "read", "key": "to_delete"})
        self.assertIsNone(read_result)

        # Try deleting again (should indicate key not found)
        delete_result_again = scratchpad_tool.invoke(
            {"action": "delete", "key": "to_delete"}
        )
        self.assertEqual(
            delete_result_again, {"status": "key_not_found", "key": "to_delete"}
        )

    def test_clear_all(self):
        """Test clearing all keys from the scratchpad."""
        # Write some keys
        scratchpad_tool.invoke({"action": "write", "key": "key_1", "value": "val_1"})
        scratchpad_tool.invoke({"action": "write", "key": "key_2", "value": "val_2"})

        # Check list is not empty
        keys = scratchpad_tool.invoke({"action": "list"})
        self.assertCountEqual(keys, ["key_1", "key_2"])

        # Clear all
        clear_result = scratchpad_tool.invoke({"action": "clear_all"})
        self.assertEqual(clear_result, {"status": "success"})

        # Check list is empty
        keys_after_clear = scratchpad_tool.invoke({"action": "list"})
        self.assertEqual(keys_after_clear, [])

        # Check keys are gone by trying to read them
        read_result_1 = scratchpad_tool.invoke({"action": "read", "key": "key_1"})
        self.assertIsNone(read_result_1)
        read_result_2 = scratchpad_tool.invoke({"action": "read", "key": "key_2"})
        self.assertIsNone(read_result_2)

    def test_invalid_action_or_args(self):
        """Test API behavior with invalid actions or missing arguments."""
        # Test invalid action - Expecting an error dictionary from the API
        result_invalid = scratchpad_tool.invoke({"action": "invalid_action"})
        self.assertIsInstance(
            result_invalid, dict, "Result should be a dict for invalid action"
        )
        # Check for common error indicators in the API response
        self.assertTrue(
            result_invalid.get("status") == "error"
            or "invalid" in result_invalid.get("message", "").lower(),
            f"Expected error status or message for invalid action, got: {result_invalid}",
        )

        # Test missing key for write - Expecting an error dictionary
        result_write_no_key = scratchpad_tool.invoke(
            {"action": "write", "value": "some_value"}
        )
        self.assertIsInstance(
            result_write_no_key,
            dict,
            "Result should be a dict for missing key in write",
        )
        self.assertTrue(
            result_write_no_key.get("status") == "error"
            or "missing" in result_write_no_key.get("message", "").lower()
            or "required" in result_write_no_key.get("message", "").lower(),
            f"Expected error status or message for missing key in write, got: {result_write_no_key}",
        )

        # Test missing value for write - Expecting an error dictionary
        result_write_no_value = scratchpad_tool.invoke(
            {"action": "write", "key": "some_key"}
        )
        self.assertIsInstance(
            result_write_no_value,
            dict,
            "Result should be a dict for missing value in write",
        )
        self.assertTrue(
            result_write_no_value.get("status") == "error"
            or "missing" in result_write_no_value.get("message", "").lower()
            or "required" in result_write_no_value.get("message", "").lower(),
            f"Expected error status or message for missing value in write, got: {result_write_no_value}",
        )

        # Test missing key for read - Expecting an error dictionary
        result_read_no_key = scratchpad_tool.invoke({"action": "read"})
        self.assertIsInstance(
            result_read_no_key, dict, "Result should be a dict for missing key in read"
        )
        self.assertTrue(
            result_read_no_key.get("status") == "error"
            or "missing" in result_read_no_key.get("message", "").lower()
            or "required" in result_read_no_key.get("message", "").lower(),
            f"Expected error status or message for missing key in read, got: {result_read_no_key}",
        )

        # Test missing key for delete - Expecting an error dictionary
        result_delete_no_key = scratchpad_tool.invoke({"action": "delete"})
        self.assertIsInstance(
            result_delete_no_key,
            dict,
            "Result should be a dict for missing key in delete",
        )
        self.assertTrue(
            result_delete_no_key.get("status") == "error"
            or "missing" in result_delete_no_key.get("message", "").lower()
            or "required" in result_delete_no_key.get("message", "").lower(),
            f"Expected error status or message for missing key in delete, got: {result_delete_no_key}",
        )


if __name__ == "__main__":
    unittest.main()
