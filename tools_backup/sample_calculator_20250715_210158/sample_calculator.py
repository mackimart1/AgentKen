"""
Sample tool for Enhanced ToolMaker demonstration.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    operation: str = Field(description="Operation to perform: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

@tool(args_schema=CalculatorInput)
def sample_calculator(operation: str, a: float, b: float) -> Dict[str, Any]:
    """
    A simple calculator tool for demonstration.
    
    This tool performs basic arithmetic operations on two numbers.
    
    Args:
        operation (str): The operation to perform (add, subtract, multiply, divide)
        a (float): The first number
        b (float): The second number
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
        
    Raises:
        ValueError: When operation is not supported or division by zero
        
    Examples:
        >>> result = sample_calculator.invoke({"operation": "add", "a": 5, "b": 3})
        >>> print(result["result"])
        8.0
        
        >>> result = sample_calculator.invoke({"operation": "multiply", "a": 4, "b": 7})
        >>> print(result["result"])
        28.0
    """
    try:
        # Input validation
        if operation not in ["add", "subtract", "multiply", "divide"]:
            raise ValueError(f"Unsupported operation: {operation}")
        
        if operation == "divide" and b == 0:
            raise ValueError("Cannot divide by zero")
        
        logger.info(f"Performing {operation} on {a} and {b}")
        
        # Perform calculation
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b
        
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully performed {operation}: {a} {operation} {b} = {result}"
        }
        
    except Exception as e:
        logger.error(f"Error in sample_calculator: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Calculation failed: {str(e)}"
        }
