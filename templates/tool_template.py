"""
Tool Template System

Provides templates and utilities for creating new tools with consistent
structure, documentation, and best practices.
"""

import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime


class ToolTemplate:
    """Template for creating new tools with consistent structure"""

    @staticmethod
    def generate_tool_code(
        tool_name: str,
        description: str,
        parameters: Dict[str, str],
        return_type: str,
        author: str,
        user_id: str,
    ) -> str:
        """Generate tool code from template"""

        # Convert tool_name to snake_case for filename
        tool_filename = tool_name.lower().replace(" ", "_").replace("-", "_")

        # Generate parameter documentation
        param_docs = ToolTemplate._generate_parameter_docs(parameters)

        # Generate function signature
        func_signature = ToolTemplate._generate_function_signature(
            tool_filename, parameters, return_type
        )

        # Generate parameter validation
        param_validation = ToolTemplate._generate_parameter_validation(parameters)

        template = f'''"""
{tool_name}: {description}

Created by: {author}
User ID: {user_id}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This tool follows the standard AgentK pattern and integrates with the
permissioned creation system.
"""
from typing import Dict, Any, Optional
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
{func_signature}:
    """{description}
    
{param_docs}
    
    Returns:
        {return_type}: {description}
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If the tool execution fails
    """
    try:
        logger.info(f"Executing {tool_filename} with parameters: {{locals()}}")
        
        # Parameter validation
{param_validation}
        
        # Tool implementation
        result = _execute_{tool_filename}({', '.join(parameters.keys())})
        
        logger.info(f"{tool_filename} completed successfully")
        return result
        
    except ValueError as e:
        logger.error(f"{tool_filename} parameter error: {{e}}")
        raise
    except Exception as e:
        logger.error(f"{tool_filename} execution error: {{e}}")
        raise RuntimeError(f"Tool execution failed: {{str(e)}}")

def _execute_{tool_filename}({', '.join(parameters.keys())}) -> {return_type}:
    """
    Internal implementation of {tool_name}.
    
    Args:
{param_docs}
        
    Returns:
        {return_type}: {description}
    """
    # TODO: Implement the actual tool logic here
    # This is a placeholder implementation
    
    # Example implementation - replace with actual logic
    if not any([{', '.join(parameters.keys())}]):
        raise ValueError("At least one parameter must be provided")
    
    # Placeholder result - replace with actual implementation
    result = f"{tool_name} executed with: {{', '.join([f'{{k}}={{v}}' for k, v in locals().items() if k != 'result'])}}"
    
    return result

def validate_{tool_filename}_input(**kwargs) -> bool:
    """
    Validate input parameters for {tool_name}.
    
    Args:
        **kwargs: The input parameters to validate
        
    Returns:
        bool: True if parameters are valid, False otherwise
    """
    try:
        # Add validation logic here
        required_params = {list(parameters.keys())}
        
        for param in required_params:
            if param not in kwargs:
                logger.warning(f"Missing required parameter: {{param}}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {{e}}")
        return False

if __name__ == "__main__":
    # Test the tool
    test_params = {{
{chr(10).join([f'        "{param}": {ToolTemplate._get_test_value(param_type)},' for param, param_type in parameters.items()])}
    }}
    
    try:
        result = {tool_filename}(**test_params)
        print(f"Test result: {{result}}")
    except Exception as e:
        print(f"Test failed: {{e}}")
'''

        return template

    @staticmethod
    def _generate_parameter_docs(parameters: Dict[str, str]) -> str:
        """Generate parameter documentation string"""
        docs = []
        for param_name, param_type in parameters.items():
            docs.append(f"    {param_name} ({param_type}): Description of {param_name}")

        return "\n".join(docs)

    @staticmethod
    def _generate_function_signature(
        tool_name: str, parameters: Dict[str, str], return_type: str
    ) -> str:
        """Generate function signature with proper typing"""
        param_signature = []
        for param_name, param_type in parameters.items():
            param_signature.append(f"{param_name}: {param_type}")

        signature = f"def {tool_name}({', '.join(param_signature)}) -> {return_type}"
        return signature

    @staticmethod
    def _generate_parameter_validation(parameters: Dict[str, str]) -> str:
        """Generate parameter validation code"""
        validation_lines = []

        for param_name, param_type in parameters.items():
            if param_type == "str":
                validation_lines.append(
                    f"        if not {param_name} or not isinstance({param_name}, str):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be a non-empty string")'
                )
            elif param_type == "int":
                validation_lines.append(
                    f"        if not isinstance({param_name}, int):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be an integer")'
                )
            elif param_type == "float":
                validation_lines.append(
                    f"        if not isinstance({param_name}, (int, float)):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be a number")'
                )
            elif param_type == "bool":
                validation_lines.append(
                    f"        if not isinstance({param_name}, bool):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be a boolean")'
                )
            elif param_type == "Dict[str, Any]":
                validation_lines.append(
                    f"        if not isinstance({param_name}, dict):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be a dictionary")'
                )
            elif param_type == "List[str]":
                validation_lines.append(
                    f"        if not isinstance({param_name}, list):"
                )
                validation_lines.append(
                    f'            raise ValueError(f"{param_name} must be a list")'
                )

        return "\n".join(validation_lines)

    @staticmethod
    def generate_test_code(
        tool_name: str, parameters: Dict[str, str], return_type: str
    ) -> str:
        """Generate test code for the tool"""

        tool_filename = tool_name.lower().replace(" ", "_").replace("-", "_")

        # Generate test parameters
        test_params = ToolTemplate._generate_test_parameters(parameters)

        template = f'''"""
Tests for {tool_name} tool

Generated by the permissioned creation system.
"""
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.{tool_filename} import {tool_filename}, validate_{tool_filename}_input

class Test{tool_name.replace(' ', '').replace('-', '')}(unittest.TestCase):
    """Test cases for {tool_name} tool"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_tool_creation(self):
        """Test that the tool can be created and called"""
        test_params = {{
{chr(10).join([f'            "{param}": "test_value"' for param in parameters.keys()])}
        }}
        
        result = {tool_filename}(**test_params)
        
        self.assertIsInstance(result, {return_type})
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        valid_params = {{
{chr(10).join([f'            "{param}": "test_value"' for param in parameters.keys()])}
        }}
        
        self.assertTrue(validate_{tool_filename}_input(**valid_params))
        
        # Test invalid parameters (missing required)
        invalid_params = {{}}
        self.assertFalse(validate_{tool_filename}_input(**invalid_params))
    
    def test_successful_execution(self):
        """Test successful tool execution"""
        test_params = {{
{chr(10).join([f'            "{param}": "test_value"' for param in parameters.keys()])}
        }}
        
        result = {tool_filename}(**test_params)
        
        self.assertIsInstance(result, {return_type})
        self.assertIsNotNone(result)
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        # Test with empty parameters
        with self.assertRaises(ValueError):
            {tool_filename}()
        
        # Test with invalid parameter types
        invalid_params = {{
{chr(10).join([f'            "{param}": None' for param in parameters.keys()])}
        }}
        
        with self.assertRaises((ValueError, RuntimeError)):
            {tool_filename}(**invalid_params)
    
    def test_tool_decorator(self):
        """Test that the tool is properly decorated"""
        # Check if the tool has the expected attributes from @tool decorator
        self.assertTrue(hasattr({tool_filename}, 'name'))
        self.assertTrue(hasattr({tool_filename}, 'description'))
        self.assertTrue(hasattr({tool_filename}, 'args_schema'))
    
    def test_logging(self):
        """Test that the tool logs appropriately"""
        with self.assertLogs(level='INFO') as log:
            test_params = {{
{chr(10).join([f'                "{param}": "test_value"' for param in parameters.keys()])}
            }}
            
            {tool_filename}(**test_params)
            
            # Check that logs were generated
            self.assertTrue(len(log.records) > 0)
    
    def test_return_type(self):
        """Test that the tool returns the expected type"""
        test_params = {{
{chr(10).join([f'            "{param}": "test_value"' for param in parameters.keys()])}
        }}
        
        result = {tool_filename}(**test_params)
        
        # Check return type
        if {return_type} == "str":
            self.assertIsInstance(result, str)
        elif {return_type} == "int":
            self.assertIsInstance(result, int)
        elif {return_type} == "float":
            self.assertIsInstance(result, (int, float))
        elif {return_type} == "bool":
            self.assertIsInstance(result, bool)
        elif {return_type} == "Dict[str, Any]":
            self.assertIsInstance(result, dict)
        elif {return_type} == "List[str]":
            self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()
'''

        return template

    @staticmethod
    def _get_test_value(param_type: str) -> str:
        """Get appropriate test value for a parameter type"""
        if param_type == "str":
            return '"test_string"'
        elif param_type == "int":
            return "42"
        elif param_type == "float":
            return "3.14"
        elif param_type == "bool":
            return "True"
        elif param_type == "Dict[str, Any]":
            return '{"key": "value"}'
        elif param_type == "List[str]":
            return '["item1", "item2"]'
        else:
            return '"test_value"'

    @staticmethod
    def _generate_test_parameters(parameters: Dict[str, str]) -> str:
        """Generate test parameter values"""
        test_values = []
        for param_name, param_type in parameters.items():
            if param_type == "str":
                test_values.append(f'"{param_name}": "test_string"')
            elif param_type == "int":
                test_values.append(f'"{param_name}": 42')
            elif param_type == "float":
                test_values.append(f'"{param_name}": 3.14')
            elif param_type == "bool":
                test_values.append(f'"{param_name}": True')
            elif param_type == "Dict[str, Any]":
                test_values.append(f'"{param_name}": {{"key": "value"}}')
            elif param_type == "List[str]":
                test_values.append(f'"{param_name}": ["item1", "item2"]')
            else:
                test_values.append(f'"{param_name}": "test_value"')

        return "\n".join(test_values)

    @staticmethod
    def generate_manifest_entry(
        tool_name: str,
        description: str,
        parameters: Dict[str, str],
        return_type: str,
        author: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Generate manifest entry for the tool"""
        tool_filename = tool_name.lower().replace(" ", "_").replace("-", "_")
        return {
            "name": tool_filename,
            "module_path": f"tools/{tool_filename}.py",
            "function_name": tool_filename,
            "description": description,
            "parameters": parameters,
            "return_type": return_type,
            "author": author,
            "created_by": user_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "status": "active",
        }
