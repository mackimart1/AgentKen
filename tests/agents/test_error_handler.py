import unittest
from agents.error_handler import error_handler_agent


class TestErrorHandlerAgent(unittest.TestCase):

    def test_file_not_found_error(self):
        error_message = "Error: File not found: /path/to/nonexistent_file.txt"
        result = error_handler_agent.invoke({"error_message": error_message})
        self.assertIn("File or directory not found", result["analysis"])
        self.assertIn("Verify the file path", result["suggested_fix"])

    def test_permission_denied_error(self):
        error_message = "Permission denied: /var/log/syslog"
        result = error_handler_agent.invoke({"error_message": error_message})
        self.assertIn("Insufficient permissions", result["analysis"])
        self.assertIn("Check file/directory permissions", result["suggested_fix"])

    def test_command_not_found_error(self):
        error_message = "bash: some_command: command not found"
        result = error_handler_agent.invoke({"error_message": error_message})
        self.assertIn("Command not found in system PATH", result["analysis"])
        self.assertIn("Ensure the command is installed", result["suggested_fix"])

    def test_syntax_error(self):
        error_message = "SyntaxError: invalid syntax"
        result = error_handler_agent.invoke({"error_message": error_message})
        self.assertIn("Syntax error in code or command", result["analysis"])
        self.assertIn("Review the syntax", result["suggested_fix"])

    def test_unknown_error(self):
        error_message = "Some unexpected error occurred."
        result = error_handler_agent.invoke({"error_message": error_message})
        self.assertIn("Unknown error", result["analysis"])
        self.assertIn("Please provide more context", result["suggested_fix"])


if __name__ == "__main__":
    unittest.main()
