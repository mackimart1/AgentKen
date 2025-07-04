import unittest
import os
import shutil
import tempfile

# Adjust import path based on actual project structure if needed
# Assuming tools are importable from a 'tools' package/directory
from tools import list_directory


class TestListDirectory(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="test_list_dir_")
        # Create some files and subdirectories within the test directory
        with open(os.path.join(self.test_dir, "file1.txt"), "w") as f:
            f.write("test")
        os.makedirs(os.path.join(self.test_dir, "subdir1"))
        with open(os.path.join(self.test_dir, "subdir1", "file2.txt"), "w") as f:
            f.write("test2")
        # Add a hidden file
        with open(os.path.join(self.test_dir, ".hiddenfile"), "w") as f:
            f.write("hidden")

    def tearDown(self):
        # Remove the temporary directory and its contents after the test
        # Add error handling in case the directory was already removed or never created
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_list_directory_success(self):
        """Test listing contents of an existing directory."""
        # os.listdir includes hidden files by default
        expected_contents = sorted(["file1.txt", "subdir1", ".hiddenfile"])
        # Invoke the tool function directly for testing
        # Ensure the tool's invoke method handles the input dictionary correctly
        result = list_directory.list_directory.invoke({"directory_path": self.test_dir})
        actual_contents = sorted(result)
        self.assertEqual(actual_contents, expected_contents)

    def test_list_directory_nonexistent(self):
        """Test listing contents of a non-existent directory."""
        non_existent_path = os.path.join(self.test_dir, "non_existent_dir")
        with self.assertRaisesRegex(ValueError, "Error: Directory not found"):
            list_directory.list_directory.invoke({"directory_path": non_existent_path})

    def test_list_directory_is_file(self):
        """Test listing contents when path is a file."""
        file_path = os.path.join(self.test_dir, "file1.txt")
        with self.assertRaisesRegex(ValueError, "Error: Directory not found"):
            list_directory.list_directory.invoke({"directory_path": file_path})

    def test_list_empty_directory(self):
        """Test listing an empty directory."""
        empty_dir = os.path.join(self.test_dir, "empty_subdir")
        os.makedirs(empty_dir)
        expected_contents = []
        actual_contents = list_directory.list_directory.invoke(
            {"directory_path": empty_dir}
        )
        self.assertEqual(actual_contents, expected_contents)


if __name__ == "__main__":
    unittest.main()
