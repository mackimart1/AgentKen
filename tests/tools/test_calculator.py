import unittest
import sys
import os

# Add the project root directory (parent of 'tools' and 'tests') to the Python path
# This allows importing 'tools.calculator' from the 'tests/tools' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Check if the calculator module exists before importing
tool_module_path = os.path.join(os.path.dirname(__file__), "../../tools/calculator.py")
if not os.path.exists(tool_module_path):
    # If the file doesn't exist (e.g., during test discovery before agent runs),
    # create dummy dirs/file to prevent premature ImportError.
    # The actual agent run should overwrite this with the real code.
    print(
        f"Warning: Tool file not found at {tool_module_path}. Creating dummy file for import.",
        file=sys.stderr,
    )
    os.makedirs(os.path.dirname(tool_module_path), exist_ok=True)
    if not os.path.exists(tool_module_path):
        with open(tool_module_path, "w") as f:
            f.write("# Dummy file for import\n")
            f.write("from langchain_core.tools import tool\n")
            f.write("@tool\ndef calculator(): pass\n")  # Minimal dummy content

# Now attempt the import
try:
    # Import the specific tool function decorated by @tool
    from tools.calculator import calculator as calculator_tool_func
except ImportError as e:
    # If import still fails, raise a more informative error
    raise ImportError(
        f"Could not import calculator tool from '{tool_module_path}'. Check file structure, sys.path, and potential errors in the tool file itself. Error: {e}"
    )


class TestCalculator(unittest.TestCase):

    def test_addition(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "2 + 3"}), 5.0)
        self.assertEqual(
            calculator_tool_func.invoke({"expression": " 2+3 "}), 5.0
        )  # Test with spaces

    def test_subtraction(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "10 - 4"}), 6.0)

    def test_multiplication(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "5 * 6"}), 30.0)

    def test_division(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "10 / 2"}), 5.0)

    def test_float_division(self):
        self.assertAlmostEqual(
            calculator_tool_func.invoke({"expression": "10 / 3"}), 3.3333333333333335
        )

    def test_order_of_operations(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "2 + 3 * 4"}), 14.0)
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "(2 + 3) * 4"}), 20.0
        )

    def test_division_by_zero(self):
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "1 / 0"}),
            "Error: Division by zero.",
        )
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "1 / (2 - 2)"}),
            "Error: Division by zero.",
        )

    def test_invalid_expression_syntax(self):
        result = calculator_tool_func.invoke({"expression": "2 + * 3"})
        self.assertTrue(result.startswith("Invalid syntax:"))
        result_paren = calculator_tool_func.invoke(
            {"expression": "(2 + 3"}
        )  # Missing closing parenthesis
        self.assertTrue(result_paren.startswith("Invalid syntax:"))
        result_op = calculator_tool_func.invoke(
            {"expression": "2+"}
        )  # Incomplete expression
        self.assertTrue(result_op.startswith("Invalid syntax:"))

    def test_invalid_expression_name(self):
        # Test disallowed names (should raise NameError from _safe_eval -> ValueError in tool)
        result_os = calculator_tool_func.invoke(
            {"expression": "os.system('echo hello')"}
        )
        self.assertTrue(
            "Use of disallowed name 'os'" in result_os
            or "Invalid expression: NameError" in result_os
        )

        result_print = calculator_tool_func.invoke({"expression": "print('hello')"})
        self.assertTrue(
            "Use of disallowed name 'print'" in result_print
            or "Invalid expression: NameError" in result_print
        )

        # Test disallowed builtins (e.g. __import__) - should be caught by invalid char check first
        result_import = calculator_tool_func.invoke(
            {"expression": "__import__('os').system('echo hello')"}
        )
        self.assertTrue("Invalid characters (__)" in result_import)

        # Test using allowed names incorrectly (should raise TypeError from eval -> ValueError in tool)
        result_type_error = calculator_tool_func.invoke({"expression": "float + 1"})
        self.assertTrue(result_type_error.startswith("Invalid operation:"))

    def test_complex_expression(self):
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "100 / (2 + 3) * 2 - 1"}), 39.0
        )
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "100 / ((1+1) + 3) * 2 - 1"}),
            39.0,
        )

    def test_negative_numbers(self):
        self.assertEqual(calculator_tool_func.invoke({"expression": "-5 + 10"}), 5.0)
        self.assertEqual(calculator_tool_func.invoke({"expression": "10 * -2"}), -20.0)
        self.assertEqual(calculator_tool_func.invoke({"expression": "-10 / -2"}), 5.0)
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "10 + (-2)"}), 8.0
        )  # Parentheses with negative

    def test_invalid_chars(self):
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "1;print(1)"}),
            "Error: Invalid characters (;) in expression.",
        )
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "1+__import__('os')"}),
            "Error: Invalid characters (__) in expression.",
        )

    def test_empty_input(self):
        result = calculator_tool_func.invoke({"expression": ""})
        # compile('') raises SyntaxError: unexpected EOF while parsing
        self.assertTrue(result.startswith("Invalid syntax:"))
        result_space = calculator_tool_func.invoke({"expression": "   "})
        self.assertTrue(result_space.startswith("Invalid syntax:"))

    def test_large_numbers(self):
        # Python handles large integers automatically
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "100000000000000000000 * 2"}),
            2e21,
        )
        # Test with underscores (allowed by compile/eval if Python version supports PEP 515)
        # self.assertEqual(calculator_tool_func.invoke({"expression": "1_000_000 + 1"}), 1000001.0) # This might fail if eval context doesn't support underscores fully

    def test_spaces(self):
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "  2 +   3  "}), 5.0
        )
        self.assertEqual(
            calculator_tool_func.invoke({"expression": "5*6"}), 30.0
        )  # No spaces


if __name__ == "__main__":
    # Ensure the test runner can find the tests in this file
    unittest.main()
