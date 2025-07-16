from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import subprocess
import tempfile
import ast
import re
from pathlib import Path
from datetime import datetime


class GenerateTestsInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to generate tests for")
    agent_file_path: str = Field(description="Path to the agent file")
    test_types: List[str] = Field(
        default=["unit", "integration", "performance"], 
        description="Types of tests to generate"
    )
    custom_test_cases: List[Dict[str, Any]] = Field(
        default=[], 
        description="Custom test cases to include"
    )


class RunTestsInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to test")
    test_file_path: Optional[str] = Field(
        default=None, 
        description="Path to test file (auto-detected if not provided)"
    )
    test_types: List[str] = Field(
        default=[], 
        description="Specific test types to run (empty for all)"
    )
    timeout: int = Field(default=120, description="Test timeout in seconds")


class ValidateAgentInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to validate")
    agent_file_path: str = Field(description="Path to the agent file")
    validation_level: str = Field(
        default="standard", 
        description="Validation level: basic, standard, comprehensive"
    )


# Global test results storage
_test_results: Dict[str, Dict[str, Any]] = {}
_test_templates = {}


def _analyze_agent_code(file_path: str) -> Dict[str, Any]:
    """Analyze agent code to understand its structure and dependencies."""
    analysis = {
        "imports": [],
        "functions": [],
        "classes": [],
        "tools_used": [],
        "has_main_function": False,
        "return_format": "unknown",
        "error_handling": False
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith("tools."):
                        for alias in node.names:
                            analysis["tools_used"].append(alias.name)
                    analysis["imports"].append(node.module)
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append(node.name)
                
                # Check if this is the main agent function
                if node.name.replace("_", "").lower() in file_path.lower():
                    analysis["has_main_function"] = True
                    
                    # Analyze return statements
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value:
                            if isinstance(child.value, ast.Dict):
                                analysis["return_format"] = "dict"
                        elif isinstance(child, ast.Try):
                            analysis["error_handling"] = True
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append(node.name)
        
        # Check for common patterns
        if "logging" in content:
            analysis["has_logging"] = True
        if "Dict[str, Any]" in content:
            analysis["return_format"] = "dict"
        
    except Exception as e:
        analysis["analysis_error"] = str(e)
    
    return analysis


def _generate_unit_tests(agent_name: str, analysis: Dict[str, Any]) -> List[str]:
    """Generate unit test cases."""
    tests = []
    
    # Basic functionality test
    tests.append(f'''
    def test_{agent_name}_basic_functionality(self):
        """Test basic agent functionality."""
        from agents.{agent_name} import {agent_name}
        
        # Test with simple task
        result = {agent_name}("test task")
        
        # Verify return format
        self.assertIsInstance(result, dict, "Agent should return a dictionary")
        self.assertIn('status', result, "Result should contain 'status' field")
        self.assertIn('result', result, "Result should contain 'result' field")
        self.assertIn('message', result, "Result should contain 'message' field")
        
        # Verify status values
        self.assertIn(result['status'], ['success', 'failure'], 
                     "Status should be 'success' or 'failure'")
''')
    
    # Input validation test
    tests.append(f'''
    def test_{agent_name}_input_validation(self):
        """Test agent input validation."""
        from agents.{agent_name} import {agent_name}
        
        # Test with empty string
        result = {agent_name}("")
        self.assertIsInstance(result, dict)
        
        # Test with None (should handle gracefully)
        try:
            result = {agent_name}(None)
            self.assertIsInstance(result, dict)
        except Exception as e:
            # Some agents may raise exceptions for None input, which is acceptable
            self.assertIsInstance(e, (TypeError, ValueError))
        
        # Test with very long string
        long_task = "x" * 10000
        result = {agent_name}(long_task)
        self.assertIsInstance(result, dict)
''')
    
    # Error handling test
    if analysis.get("error_handling", False):
        tests.append(f'''
    def test_{agent_name}_error_handling(self):
        """Test agent error handling capabilities."""
        from agents.{agent_name} import {agent_name}
        from unittest.mock import patch
        
        # Test with task that might cause errors
        result = {agent_name}("invalid task that should cause error")
        self.assertIsInstance(result, dict)
        
        # If it fails, it should fail gracefully
        if result['status'] == 'failure':
            self.assertIsInstance(result['message'], str)
            self.assertTrue(len(result['message']) > 0)
''')
    
    # Tool integration test
    if analysis.get("tools_used"):
        tools_list = ", ".join(analysis["tools_used"])
        tests.append(f'''
    def test_{agent_name}_tool_integration(self):
        """Test agent tool integration."""
        from agents.{agent_name} import {agent_name}
        
        # Test that agent can handle tool-related tasks
        # Tools used: {tools_list}
        result = {agent_name}("task that requires tools")
        
        self.assertIsInstance(result, dict)
        # Should not crash even if tools are not available
''')
    
    return tests


def _generate_integration_tests(agent_name: str, analysis: Dict[str, Any]) -> List[str]:
    """Generate integration test cases."""
    tests = []
    
    # System integration test
    tests.append(f'''
    def test_{agent_name}_system_integration(self):
        """Test agent integration with the system."""
        from agents.{agent_name} import {agent_name}
        import sys
        import os
        
        # Ensure agent can be imported and called
        result = {agent_name}("system integration test")
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        
        # Test that agent doesn't interfere with system state
        original_path = sys.path.copy()
        original_cwd = os.getcwd()
        
        result = {agent_name}("test system state preservation")
        
        self.assertEqual(sys.path, original_path, "Agent should not modify sys.path")
        self.assertEqual(os.getcwd(), original_cwd, "Agent should not change working directory")
''')
    
    # Memory usage test
    tests.append(f'''
    def test_{agent_name}_memory_usage(self):
        """Test agent memory usage."""
        from agents.{agent_name} import {agent_name}
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run agent multiple times
        for i in range(5):
            result = {agent_name}(f"memory test iteration {{i}}")
            self.assertIsInstance(result, dict)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024, 
                       "Agent should not cause excessive memory usage")
''')
    
    return tests


def _generate_performance_tests(agent_name: str, analysis: Dict[str, Any]) -> List[str]:
    """Generate performance test cases."""
    tests = []
    
    # Response time test
    tests.append(f'''
    def test_{agent_name}_response_time(self):
        """Test agent response time."""
        from agents.{agent_name} import {agent_name}
        import time
        
        start_time = time.time()
        result = {agent_name}("performance test task")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        self.assertIsInstance(result, dict)
        self.assertLess(execution_time, 30, 
                       f"Agent should complete within 30 seconds, took {{execution_time:.2f}}s")
        
        # If result includes execution_time, verify it's reasonable
        if 'execution_time' in result:
            self.assertIsInstance(result['execution_time'], (int, float))
            self.assertGreater(result['execution_time'], 0)
''')
    
    # Concurrent execution test
    tests.append(f'''
    def test_{agent_name}_concurrent_execution(self):
        """Test agent concurrent execution."""
        from agents.{agent_name} import {agent_name}
        import threading
        import time
        
        results = []
        errors = []
        
        def run_agent(task_id):
            try:
                result = {agent_name}(f"concurrent test {{task_id}}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 3 concurrent instances
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_agent, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Concurrent execution should not cause errors: {{errors}}")
        self.assertEqual(len(results), 3, "All concurrent executions should complete")
        
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)
''')
    
    return tests


@tool(args_schema=GenerateTestsInput)
def generate_agent_tests(
    agent_name: str,
    agent_file_path: str,
    test_types: List[str] = None,
    custom_test_cases: List[Dict[str, Any]] = None
) -> str:
    """
    Generate comprehensive test suite for an agent.
    
    Args:
        agent_name: Name of the agent to generate tests for
        agent_file_path: Path to the agent file
        test_types: Types of tests to generate
        custom_test_cases: Custom test cases to include
    
    Returns:
        JSON string with test generation result
    """
    try:
        if test_types is None:
            test_types = ["unit", "integration", "performance"]
        if custom_test_cases is None:
            custom_test_cases = []
        
        # Validate agent file exists
        if not os.path.exists(agent_file_path):
            return json.dumps({
                "status": "failure",
                "message": f"Agent file not found: {agent_file_path}"
            })
        
        # Analyze agent code
        analysis = _analyze_agent_code(agent_file_path)
        
        # Generate test cases
        all_tests = []
        
        if "unit" in test_types:
            all_tests.extend(_generate_unit_tests(agent_name, analysis))
        
        if "integration" in test_types:
            all_tests.extend(_generate_integration_tests(agent_name, analysis))
        
        if "performance" in test_types:
            all_tests.extend(_generate_performance_tests(agent_name, analysis))
        
        # Add custom test cases
        for custom_test in custom_test_cases:
            if "test_function" in custom_test:
                all_tests.append(custom_test["test_function"])
        
        # Create test file content
        test_content = f'''"""
Comprehensive test suite for {agent_name} agent.
Generated automatically by Enhanced Agent Smith.

Test Types: {", ".join(test_types)}
Generated: {datetime.now().isoformat()}
"""

import unittest
import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Test{agent_name.title().replace("_", "")}(unittest.TestCase):
    """Comprehensive test cases for {agent_name} agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent_name = "{agent_name}"
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_agent_import(self):
        """Test that agent can be imported successfully."""
        try:
            from agents.{agent_name} import {agent_name}
            self.assertTrue(callable({agent_name}), "Agent should be callable")
        except ImportError as e:
            self.fail(f"Failed to import agent: {{e}}")
'''
        
        # Add generated tests
        for test in all_tests:
            test_content += test + "\n"
        
        # Add test runner
        test_content += '''

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
'''
        
        # Create test file
        test_dir = Path("tests/agents")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / f"test_{agent_name}.py"
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        return json.dumps({
            "status": "success",
            "test_file_path": str(test_file_path),
            "test_types": test_types,
            "total_tests": len(all_tests) + 1,  # +1 for import test
            "analysis": analysis,
            "message": f"Generated {len(all_tests) + 1} tests for {agent_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to generate tests: {str(e)}"
        })


@tool(args_schema=RunTestsInput)
def run_agent_tests(
    agent_name: str,
    test_file_path: Optional[str] = None,
    test_types: List[str] = None,
    timeout: int = 120
) -> str:
    """
    Run tests for an agent and return detailed results.
    
    Args:
        agent_name: Name of the agent to test
        test_file_path: Path to test file (auto-detected if not provided)
        test_types: Specific test types to run (empty for all)
        timeout: Test timeout in seconds
    
    Returns:
        JSON string with test results
    """
    try:
        if test_types is None:
            test_types = []
        
        # Auto-detect test file if not provided
        if not test_file_path:
            test_file_path = f"tests/agents/test_{agent_name}.py"
        
        if not os.path.exists(test_file_path):
            return json.dumps({
                "status": "failure",
                "message": f"Test file not found: {test_file_path}"
            })
        
        # Prepare test command
        cmd = [
            "python", "-m", "unittest", 
            f"tests.agents.test_{agent_name}", "-v"
        ]
        
        # Run tests
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            execution_time = time.time() - start_time
            
            # Parse test output
            test_results = _parse_unittest_output(result.stdout, result.stderr)
            
            # Store results
            _test_results[agent_name] = {
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "return_code": result.returncode,
                "results": test_results
            }
            
            overall_status = "success" if result.returncode == 0 else "failure"
            
            return json.dumps({
                "status": overall_status,
                "agent_name": agent_name,
                "test_file": test_file_path,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_results": test_results,
                "summary": {
                    "total_tests": test_results.get("total_tests", 0),
                    "passed": test_results.get("passed", 0),
                    "failed": test_results.get("failed", 0),
                    "errors": test_results.get("errors", 0),
                    "success_rate": test_results.get("success_rate", 0.0)
                },
                "message": f"Tests completed in {execution_time:.2f}s with return code {result.returncode}"
            })
            
        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "timeout",
                "message": f"Tests timed out after {timeout} seconds",
                "agent_name": agent_name
            })
            
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to run tests: {str(e)}"
        })


def _parse_unittest_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse unittest output to extract test results."""
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "test_details": [],
        "success_rate": 0.0
    }
    
    all_output = stdout + "\n" + stderr
    lines = all_output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Count test results
        if " ... ok" in line:
            results["passed"] += 1
            results["total_tests"] += 1
            test_name = line.split(" ... ok")[0].strip()
            results["test_details"].append({
                "name": test_name,
                "status": "passed",
                "message": ""
            })
        elif " ... FAIL" in line:
            results["failed"] += 1
            results["total_tests"] += 1
            test_name = line.split(" ... FAIL")[0].strip()
            results["test_details"].append({
                "name": test_name,
                "status": "failed",
                "message": "Test failed"
            })
        elif " ... ERROR" in line:
            results["errors"] += 1
            results["total_tests"] += 1
            test_name = line.split(" ... ERROR")[0].strip()
            results["test_details"].append({
                "name": test_name,
                "status": "error",
                "message": "Test error"
            })
        elif " ... skipped" in line:
            results["skipped"] += 1
            results["total_tests"] += 1
            test_name = line.split(" ... skipped")[0].strip()
            results["test_details"].append({
                "name": test_name,
                "status": "skipped",
                "message": "Test skipped"
            })
    
    # Calculate success rate
    if results["total_tests"] > 0:
        results["success_rate"] = (results["passed"] / results["total_tests"]) * 100
    
    return results


@tool(args_schema=ValidateAgentInput)
def validate_agent_code(
    agent_name: str,
    agent_file_path: str,
    validation_level: str = "standard"
) -> str:
    """
    Validate agent code for compliance and best practices.
    
    Args:
        agent_name: Name of the agent to validate
        agent_file_path: Path to the agent file
        validation_level: Validation level (basic, standard, comprehensive)
    
    Returns:
        JSON string with validation results
    """
    try:
        if not os.path.exists(agent_file_path):
            return json.dumps({
                "status": "failure",
                "message": f"Agent file not found: {agent_file_path}"
            })
        
        validation_results = {
            "compliance_checks": [],
            "best_practices": [],
            "warnings": [],
            "errors": [],
            "score": 0,
            "max_score": 0
        }
        
        # Read agent code
        with open(agent_file_path, 'r') as f:
            code = f.read()
        
        # Analyze code structure
        analysis = _analyze_agent_code(agent_file_path)
        
        # Basic validation checks
        checks = [
            ("has_main_function", "Agent has main function", analysis.get("has_main_function", False)),
            ("return_format", "Returns dictionary format", analysis.get("return_format") == "dict"),
            ("error_handling", "Has error handling", analysis.get("error_handling", False)),
            ("has_logging", "Uses logging", analysis.get("has_logging", False)),
            ("has_docstring", "Has docstring", '"""' in code or "'''" in code),
            ("imports_typing", "Uses type hints", "typing" in str(analysis.get("imports", []))),
        ]
        
        if validation_level in ["standard", "comprehensive"]:
            checks.extend([
                ("status_field", "Returns status field", '"status"' in code or "'status'" in code),
                ("result_field", "Returns result field", '"result"' in code or "'result'" in code),
                ("message_field", "Returns message field", '"message"' in code or "'message'" in code),
                ("input_validation", "Validates input", "if not" in code or "isinstance" in code),
            ])
        
        if validation_level == "comprehensive":
            checks.extend([
                ("performance_logging", "Logs performance", "time" in code.lower()),
                ("memory_efficient", "Memory efficient", "del " in code or "gc.collect" in code),
                ("thread_safe", "Thread safe considerations", "threading" in code or "lock" in code.lower()),
                ("exception_specific", "Specific exception handling", "except " in code and "Exception" not in code),
            ])
        
        # Evaluate checks
        passed_checks = 0
        for check_id, description, passed in checks:
            validation_results["max_score"] += 1
            if passed:
                passed_checks += 1
                validation_results["compliance_checks"].append({
                    "check": description,
                    "status": "passed"
                })
            else:
                validation_results["compliance_checks"].append({
                    "check": description,
                    "status": "failed"
                })
        
        validation_results["score"] = passed_checks
        
        # Best practices checks
        best_practices = []
        
        if len(code.split('\n')) > 200:
            best_practices.append("Consider breaking large agent into smaller functions")
        
        if "print(" in code:
            best_practices.append("Use logging instead of print statements")
        
        if "time.sleep" in code:
            best_practices.append("Avoid blocking sleep calls in agents")
        
        if not re.search(r'def\s+' + agent_name + r'\s*\(', code):
            best_practices.append(f"Main function should be named '{agent_name}'")
        
        validation_results["best_practices"] = best_practices
        
        # Calculate overall grade
        score_percentage = (validation_results["score"] / validation_results["max_score"]) * 100
        
        if score_percentage >= 90:
            grade = "A"
        elif score_percentage >= 80:
            grade = "B"
        elif score_percentage >= 70:
            grade = "C"
        elif score_percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return json.dumps({
            "status": "success",
            "agent_name": agent_name,
            "validation_level": validation_level,
            "score": validation_results["score"],
            "max_score": validation_results["max_score"],
            "score_percentage": score_percentage,
            "grade": grade,
            "compliance_checks": validation_results["compliance_checks"],
            "best_practices": validation_results["best_practices"],
            "warnings": validation_results["warnings"],
            "errors": validation_results["errors"],
            "analysis": analysis,
            "message": f"Validation completed with score {validation_results['score']}/{validation_results['max_score']} ({score_percentage:.1f}%) - Grade: {grade}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to validate agent: {str(e)}"
        })


@tool
def get_test_results_summary() -> str:
    """
    Get summary of all test results.
    
    Returns:
        JSON string with test results summary
    """
    try:
        if not _test_results:
            return json.dumps({
                "status": "success",
                "message": "No test results available",
                "summary": {
                    "total_agents_tested": 0,
                    "overall_success_rate": 0.0,
                    "agents": []
                }
            })
        
        total_tests = 0
        total_passed = 0
        agent_summaries = []
        
        for agent_name, results in _test_results.items():
            test_data = results.get("results", {})
            agent_total = test_data.get("total_tests", 0)
            agent_passed = test_data.get("passed", 0)
            
            total_tests += agent_total
            total_passed += agent_passed
            
            agent_summaries.append({
                "agent_name": agent_name,
                "timestamp": results.get("timestamp", ""),
                "execution_time": results.get("execution_time", 0),
                "total_tests": agent_total,
                "passed": agent_passed,
                "failed": test_data.get("failed", 0),
                "errors": test_data.get("errors", 0),
                "success_rate": test_data.get("success_rate", 0.0)
            })
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
        
        # Sort by success rate
        agent_summaries.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return json.dumps({
            "status": "success",
            "summary": {
                "total_agents_tested": len(_test_results),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "overall_success_rate": overall_success_rate,
                "agents": agent_summaries
            },
            "message": f"Test summary for {len(_test_results)} agents with {overall_success_rate:.1f}% overall success rate"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get test summary: {str(e)}"
        })