from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import os
import ast
import subprocess
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path


class ValidateToolInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to validate")
    tool_path: str = Field(description="Path to the tool file")
    validation_level: str = Field(default="comprehensive", description="Validation level: basic, standard, comprehensive")


class RunSandboxTestInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to test")
    test_code: str = Field(description="Test code to run in sandbox")
    timeout: int = Field(default=30, description="Test timeout in seconds")


class SecurityScanInput(BaseModel):
    tool_path: str = Field(description="Path to the tool file to scan")
    scan_level: str = Field(default="standard", description="Security scan level: basic, standard, strict")


# Global validation storage
_validation_results: Dict[str, Dict[str, Any]] = {}
_sandbox_dir = Path("sandbox")
_sandbox_dir.mkdir(exist_ok=True)


def _create_isolated_sandbox(tool_name: str) -> Path:
    """Create an isolated sandbox environment for testing."""
    sandbox_path = _sandbox_dir / f"sandbox_{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sandbox_path.mkdir(exist_ok=True)
    
    # Create minimal Python environment
    init_file = sandbox_path / "__init__.py"
    init_file.touch()
    
    # Create requirements file
    requirements_file = sandbox_path / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write("langchain-core>=0.1.0\npydantic>=2.0.0\n")
    
    return sandbox_path


def _cleanup_sandbox(sandbox_path: Path):
    """Clean up sandbox environment."""
    try:
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup sandbox {sandbox_path}: {e}")


def _analyze_security_risks(code: str) -> Dict[str, Any]:
    """Analyze code for security risks."""
    risks = {
        "high_risk": [],
        "medium_risk": [],
        "low_risk": [],
        "score": 100
    }
    
    try:
        tree = ast.parse(code)
        
        # High-risk patterns
        dangerous_imports = ['os', 'subprocess', 'sys', 'eval', 'exec', 'compile', '__import__']
        dangerous_calls = ['eval', 'exec', 'compile', '__import__', 'getattr', 'setattr', 'delattr']
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_imports:
                        risks["high_risk"].append(f"Dangerous import: {alias.name} (line {node.lineno})")
                        risks["score"] -= 20
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in dangerous_imports:
                    risks["high_risk"].append(f"Dangerous import: {node.module} (line {node.lineno})")
                    risks["score"] -= 20
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_calls:
                    risks["high_risk"].append(f"Dangerous function call: {node.func.id} (line {node.lineno})")
                    risks["score"] -= 25
        
        # Medium-risk patterns
        if 'open(' in code:
            risks["medium_risk"].append("File operations detected")
            risks["score"] -= 10
        
        if any(keyword in code.lower() for keyword in ['requests', 'urllib', 'http']):
            risks["medium_risk"].append("Network operations detected")
            risks["score"] -= 5
        
        # Low-risk patterns
        if 'print(' in code:
            risks["low_risk"].append("Print statements found (consider using logging)")
        
        risks["score"] = max(0, risks["score"])
        
    except Exception as e:
        risks["high_risk"].append(f"Code analysis failed: {str(e)}")
        risks["score"] = 0
    
    return risks


def _test_tool_functionality(sandbox_path: Path, tool_name: str) -> Dict[str, Any]:
    """Test tool functionality in sandbox."""
    results = {
        "import_test": {"status": "failed", "details": "", "execution_time": 0},
        "structure_test": {"status": "failed", "details": "", "execution_time": 0},
        "basic_execution_test": {"status": "failed", "details": "", "execution_time": 0}
    }
    
    tool_file = sandbox_path / f"{tool_name}.py"
    
    # Import test
    start_time = time.time()
    try:
        result = subprocess.run([
            "python", "-c", f"import sys; sys.path.insert(0, '{sandbox_path}'); import {tool_name}"
        ], capture_output=True, text=True, timeout=10, cwd=sandbox_path)
        
        if result.returncode == 0:
            results["import_test"]["status"] = "passed"
            results["import_test"]["details"] = "Tool imports successfully"
        else:
            results["import_test"]["details"] = f"Import failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        results["import_test"]["details"] = "Import test timed out"
    except Exception as e:
        results["import_test"]["details"] = f"Import test error: {str(e)}"
    
    results["import_test"]["execution_time"] = time.time() - start_time
    
    # Structure test
    start_time = time.time()
    try:
        with open(tool_file, 'r') as f:
            code = f.read()
        
        has_tool_decorator = "@tool" in code
        has_docstring = '"""' in code or "'''" in code
        has_type_hints = ":" in code and "->" in code
        
        if has_tool_decorator and has_docstring:
            results["structure_test"]["status"] = "passed"
            results["structure_test"]["details"] = "Tool structure is valid"
        else:
            missing = []
            if not has_tool_decorator:
                missing.append("@tool decorator")
            if not has_docstring:
                missing.append("docstring")
            results["structure_test"]["details"] = f"Missing: {', '.join(missing)}"
    
    except Exception as e:
        results["structure_test"]["details"] = f"Structure test error: {str(e)}"
    
    results["structure_test"]["execution_time"] = time.time() - start_time
    
    # Basic execution test (if import succeeded)
    if results["import_test"]["status"] == "passed":
        start_time = time.time()
        try:
            test_code = f"""
import sys
sys.path.insert(0, '{sandbox_path}')
import {tool_name}

# Try to get the tool function
tool_func = getattr({tool_name}, '{tool_name}', None)
if tool_func:
    print("Tool function found")
else:
    print("Tool function not found")
"""
            
            result = subprocess.run([
                "python", "-c", test_code
            ], capture_output=True, text=True, timeout=10, cwd=sandbox_path)
            
            if result.returncode == 0 and "Tool function found" in result.stdout:
                results["basic_execution_test"]["status"] = "passed"
                results["basic_execution_test"]["details"] = "Tool function is accessible"
            else:
                results["basic_execution_test"]["details"] = f"Execution test failed: {result.stderr}"
        
        except Exception as e:
            results["basic_execution_test"]["details"] = f"Execution test error: {str(e)}"
        
        results["basic_execution_test"]["execution_time"] = time.time() - start_time
    
    return results


@tool(args_schema=ValidateToolInput)
def validate_tool_sandbox(
    tool_name: str,
    tool_path: str,
    validation_level: str = "comprehensive"
) -> str:
    """
    Validate a tool in a sandboxed environment for functionality and safety.
    
    Args:
        tool_name: Name of the tool to validate
        tool_path: Path to the tool file
        validation_level: Validation level (basic, standard, comprehensive)
    
    Returns:
        JSON string with validation results
    """
    try:
        if not os.path.exists(tool_path):
            return json.dumps({
                "status": "failure",
                "message": f"Tool file not found: {tool_path}"
            })
        
        # Create sandbox
        sandbox_path = _create_isolated_sandbox(tool_name)
        
        # Copy tool to sandbox
        tool_file = sandbox_path / f"{tool_name}.py"
        shutil.copy2(tool_path, tool_file)
        
        validation_result = {
            "tool_name": tool_name,
            "validation_level": validation_level,
            "timestamp": datetime.now().isoformat(),
            "overall_status": "failed",
            "security_analysis": {},
            "functionality_tests": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Security analysis
        with open(tool_file, 'r') as f:
            code = f.read()
        
        security_analysis = _analyze_security_risks(code)
        validation_result["security_analysis"] = security_analysis
        
        # Functionality tests
        functionality_tests = _test_tool_functionality(sandbox_path, tool_name)
        validation_result["functionality_tests"] = functionality_tests
        
        # Performance metrics
        file_size = tool_file.stat().st_size
        lines_of_code = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        validation_result["performance_metrics"] = {
            "file_size_bytes": file_size,
            "lines_of_code": lines_of_code,
            "complexity_score": min(100, lines_of_code / 2)
        }
        
        # Calculate overall status
        security_score = security_analysis["score"]
        functionality_score = sum(1 for test in functionality_tests.values() if test["status"] == "passed") / len(functionality_tests) * 100
        
        overall_score = (security_score + functionality_score) / 2
        
        if overall_score >= 80 and security_score >= 70:
            validation_result["overall_status"] = "passed"
        elif overall_score >= 60:
            validation_result["overall_status"] = "warning"
        else:
            validation_result["overall_status"] = "failed"
        
        # Generate recommendations
        if security_score < 80:
            validation_result["recommendations"].append("Review security risks and remove dangerous operations")
        
        if functionality_score < 100:
            validation_result["recommendations"].append("Fix functionality issues before deployment")
        
        if file_size > 10240:  # 10KB
            validation_result["recommendations"].append("Consider reducing file size for better performance")
        
        # Store results
        _validation_results[tool_name] = validation_result
        
        # Cleanup sandbox
        _cleanup_sandbox(sandbox_path)
        
        return json.dumps({
            "status": "success",
            "validation_result": validation_result,
            "message": f"Tool validation completed with status: {validation_result['overall_status']}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Tool validation failed: {str(e)}"
        })


@tool(args_schema=RunSandboxTestInput)
def run_sandbox_test(tool_name: str, test_code: str, timeout: int = 30) -> str:
    """
    Run custom test code for a tool in a sandboxed environment.
    
    Args:
        tool_name: Name of the tool to test
        test_code: Python test code to execute
        timeout: Test timeout in seconds
    
    Returns:
        JSON string with test results
    """
    try:
        # Create sandbox
        sandbox_path = _create_isolated_sandbox(tool_name)
        
        # Create test file
        test_file = sandbox_path / "test_custom.py"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Run test
        start_time = time.time()
        result = subprocess.run([
            "python", str(test_file)
        ], capture_output=True, text=True, timeout=timeout, cwd=sandbox_path)
        
        execution_time = time.time() - start_time
        
        test_result = {
            "tool_name": tool_name,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": "passed" if result.returncode == 0 else "failed"
        }
        
        # Cleanup sandbox
        _cleanup_sandbox(sandbox_path)
        
        return json.dumps({
            "status": "success",
            "test_result": test_result,
            "message": f"Sandbox test completed in {execution_time:.2f}s"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "failure",
            "message": f"Sandbox test timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Sandbox test failed: {str(e)}"
        })


@tool(args_schema=SecurityScanInput)
def security_scan_tool(tool_path: str, scan_level: str = "standard") -> str:
    """
    Perform security scan on a tool file.
    
    Args:
        tool_path: Path to the tool file to scan
        scan_level: Security scan level (basic, standard, strict)
    
    Returns:
        JSON string with security scan results
    """
    try:
        if not os.path.exists(tool_path):
            return json.dumps({
                "status": "failure",
                "message": f"Tool file not found: {tool_path}"
            })
        
        with open(tool_path, 'r') as f:
            code = f.read()
        
        # Perform security analysis
        security_analysis = _analyze_security_risks(code)
        
        # Additional checks based on scan level
        if scan_level in ["standard", "strict"]:
            # Check for hardcoded secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            for pattern in secret_patterns:
                import re
                if re.search(pattern, code, re.IGNORECASE):
                    security_analysis["medium_risk"].append("Potential hardcoded secrets detected")
                    security_analysis["score"] -= 15
                    break
        
        if scan_level == "strict":
            # Additional strict checks
            if 'input(' in code:
                security_analysis["medium_risk"].append("User input function detected")
                security_analysis["score"] -= 5
            
            if 'pickle' in code:
                security_analysis["high_risk"].append("Pickle usage detected (potential security risk)")
                security_analysis["score"] -= 20
        
        # Determine overall security level
        if security_analysis["score"] >= 90:
            security_level = "excellent"
        elif security_analysis["score"] >= 80:
            security_level = "good"
        elif security_analysis["score"] >= 60:
            security_level = "acceptable"
        elif security_analysis["score"] >= 40:
            security_level = "poor"
        else:
            security_level = "dangerous"
        
        return json.dumps({
            "status": "success",
            "security_analysis": security_analysis,
            "security_level": security_level,
            "scan_level": scan_level,
            "recommendations": _generate_security_recommendations(security_analysis),
            "message": f"Security scan completed - Level: {security_level}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Security scan failed: {str(e)}"
        })


def _generate_security_recommendations(security_analysis: Dict[str, Any]) -> List[str]:
    """Generate security recommendations based on analysis."""
    recommendations = []
    
    if security_analysis["high_risk"]:
        recommendations.append("CRITICAL: Remove all high-risk operations before deployment")
        recommendations.append("Consider alternative approaches that don't require dangerous functions")
    
    if security_analysis["medium_risk"]:
        recommendations.append("Review medium-risk operations and add proper validation")
        recommendations.append("Implement input sanitization and error handling")
    
    if security_analysis["score"] < 80:
        recommendations.append("Overall security score is below recommended threshold")
        recommendations.append("Consider security review before production deployment")
    
    if not recommendations:
        recommendations.append("Security analysis passed - tool appears safe for deployment")
    
    return recommendations


@tool
def get_validation_history() -> str:
    """
    Get validation history for all tools.
    
    Returns:
        JSON string with validation history
    """
    try:
        return json.dumps({
            "status": "success",
            "validation_history": _validation_results,
            "total_validations": len(_validation_results),
            "message": f"Retrieved validation history for {len(_validation_results)} tools"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get validation history: {str(e)}"
        })


@tool
def cleanup_validation_data() -> str:
    """
    Clean up old validation data and sandbox directories.
    
    Returns:
        JSON string with cleanup results
    """
    try:
        cleaned_count = 0
        
        # Clean up old sandbox directories
        if _sandbox_dir.exists():
            for item in _sandbox_dir.iterdir():
                if item.is_dir() and item.name.startswith("sandbox_"):
                    try:
                        shutil.rmtree(item)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to remove {item}: {e}")
        
        # Clear validation results older than 7 days
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 60 * 60)  # 7 days
        old_results = []
        
        for tool_name, result in _validation_results.items():
            result_time = datetime.fromisoformat(result["timestamp"]).timestamp()
            if result_time < cutoff_time:
                old_results.append(tool_name)
        
        for tool_name in old_results:
            del _validation_results[tool_name]
            cleaned_count += 1
        
        return json.dumps({
            "status": "success",
            "cleaned_items": cleaned_count,
            "message": f"Cleaned up {cleaned_count} validation items"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Cleanup failed: {str(e)}"
        })