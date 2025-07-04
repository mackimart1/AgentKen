import operator
from typing import Union
from langchain_core.tools import tool

# Define allowed operators
_ALLOWED_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

# Define allowed names (only number conversions and operators)
# Allow float and int for number parsing within the expression if needed
_ALLOWED_NAMES = {k: v for k, v in vars(__builtins__).items() if k in ["float", "int"]}
_ALLOWED_NAMES.update(_ALLOWED_OPERATORS)


def _safe_eval(expr: str):
    """
    Safely evaluates a mathematical expression string.
    Restricts available names and operators.
    """
    try:
        # Compile the expression with restricted builtins and names
        # Using mode 'eval' ensures it's an expression
        code = compile(expr, "<string>", "eval")

        # Validate names used in the expression
        # Allow only names present in _ALLOWED_NAMES or are numbers
        for name in code.co_names:
            is_number = False
            try:
                # Attempt to convert to float, handles ints and floats
                float(name)
                is_number = True
            except ValueError:
                pass  # Not a number

            if not is_number and name not in _ALLOWED_NAMES:
                raise NameError(f"Use of disallowed name '{name}'")

        # Evaluate the expression with restricted globals and locals
        # Globals contain only safe builtins (none needed here as names are checked)
        # Locals contain the allowed operators and number conversion functions
        result = eval(code, {"__builtins__": {}}, _ALLOWED_NAMES)
        return result
    except NameError as e:
        # Catch disallowed names specifically
        raise ValueError(f"Invalid expression: NameError - {e}") from e
    except ZeroDivisionError:
        raise ValueError("Error: Division by zero.") from None
    except SyntaxError as e:
        # Catch syntax errors during compile or eval
        raise ValueError(f"Invalid syntax: {e}") from e
    except TypeError as e:
        # Catch type errors during evaluation (e.g., operator misuse)
        raise ValueError(f"Invalid operation: {e}") from e
    except Exception as e:
        # Catch other potential errors during eval
        raise ValueError(f"Evaluation error: {e}") from e


@tool
def calculator(expression: str) -> Union[float, str]:
    """Evaluates a basic mathematical expression (+, -, *, /).

    Args:
        expression: A string representing the mathematical expression.
                    Example: "2 + 3 * (5 - 1)"

    Returns:
        The numerical result of the expression or an error message string.
    """
    try:
        # Basic check for potentially unsafe characters (prevent dunder, statement sep)
        if ";" in expression:
            return "Error: Invalid characters (;) in expression."
        if "__" in expression:
            return "Error: Invalid characters (__) in expression."

        result = _safe_eval(expression.strip())  # Strip whitespace
        # Ensure the result is a number before returning
        if isinstance(result, (int, float)):
            return float(result)
        else:
            # This might happen if the expression evaluates to something unexpected
            # despite the restrictions.
            return f"Error: Expression did not evaluate to a number. Result type: {type(result)}"
    except ValueError as e:
        # Catch errors raised from _safe_eval
        return str(e)
    except Exception as e:
        # Catch any unexpected errors during tool execution itself
        return f"An unexpected tool error occurred: {e}"
