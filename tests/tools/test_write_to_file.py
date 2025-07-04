import unittest
import os
import shutil

# Adjust import path if necessary
try:
    from tools.write_to_file import write_to_file
except ImportError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tools.write_to_file import write_to_file


class TestWriteToFile(unittest.TestCase):

    TEST_DIR = "temp_test_write_dir"
    TEST_FILE = os.path.join(TEST_DIR, "test_output.txt")
    TEST_SUBDIR_FILE = os.path.join(TEST_DIR, "subdir", "nested_output.log")

    def setUp(self):
        """Create a temporary directory for test files before each test."""
        # Clean up any potential leftovers from previous failed runs
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)
        # Create the test directory (don't create subdirs yet)
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        """Remove the temporary directory and its contents after each test."""
        if os.path.exists(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)

    def test_create_new_file(self):
        """Tests creating a new file with specific content."""
        content = "This is the first line.\nThis is the second line."
        path = self.TEST_FILE

        # Invoke the tool
        result = write_to_file.invoke({"path": path, "content": content})

        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            read_content = f.read()
        self.assertEqual(read_content, content)
        self.assertIn(f"Successfully wrote to {path}", result)

    def test_overwrite_existing_file(self):
        """Tests overwriting an existing file."""
        initial_content = "Initial content."
        new_content = "This content overwrites the initial content."
        path = self.TEST_FILE

        # Create initial file
        with open(path, "w", encoding="utf-8") as f:
            f.write(initial_content)

        # Invoke the tool to overwrite
        result = write_to_file.invoke({"path": path, "content": new_content})

        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            read_content = f.read()
        self.assertEqual(read_content, new_content)
        self.assertIn(f"Successfully wrote to {path}", result)

    def test_create_nested_directories(self):
        """Tests creating necessary subdirectories."""
        content = "Log entry."
        path = self.TEST_SUBDIR_FILE  # Path includes a subdirectory 'subdir'

        # Ensure the subdirectory does NOT exist initially
        self.assertFalse(os.path.exists(os.path.dirname(path)))

        # Invoke the tool
        result = write_to_file.invoke({"path": path, "content": content})

        # Check if both the directory and file were created
        self.assertTrue(os.path.exists(os.path.dirname(path)))
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            read_content = f.read()
        self.assertEqual(read_content, content)
        self.assertIn(f"Successfully wrote to {path}", result)

    # Note: Testing actual OS write errors (like permission denied) is complex
    # and often requires specific environment setup or more advanced mocking.
    # We are focusing on the tool's intended successful operation here.


if __name__ == "__main__":
    unittest.main()
