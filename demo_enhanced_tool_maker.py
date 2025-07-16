#!/usr/bin/env python3
"""
Demonstration script for Enhanced ToolMaker capabilities:
1. Tool Validation: Sandboxed testing for functionality and safety
2. Tool Documentation: Auto-generation of comprehensive documentation
3. Tool Deprecation: Complete lifecycle management with deprecation process

This script shows how the enhanced features work without requiring full tool creation.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Import the enhanced tools
from tools.tool_validation import (
    validate_tool_sandbox, run_sandbox_test, security_scan_tool,
    get_validation_history, cleanup_validation_data
)
from tools.tool_documentation import (
    generate_tool_documentation, update_tool_documentation,
    validate_tool_documentation, list_tool_documentation
)
from tools.tool_lifecycle import (
    deprecate_tool, update_tool_status, remove_tool,
    analyze_tool_usage, get_tool_lifecycle_status, cleanup_tool_lifecycle
)


def create_sample_tool():
    """Create a sample tool for demonstration purposes."""
    sample_tool_code = '''"""
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
'''
    
    # Create tools directory if it doesn't exist
    os.makedirs("tools", exist_ok=True)
    
    # Write sample tool
    tool_path = "tools/sample_calculator.py"
    with open(tool_path, 'w') as f:
        f.write(sample_tool_code)
    
    return tool_path


def demo_tool_validation():
    """Demonstrate tool validation capabilities."""
    print("=" * 80)
    print("TOOL VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample tool
    tool_path = create_sample_tool()
    tool_name = "sample_calculator"
    
    print("1. Creating sample tool for validation...")
    print(f"   Tool created: {tool_path}")
    
    print("\n2. Running comprehensive validation...")
    
    # Validate tool in sandbox
    validation_result = validate_tool_sandbox.invoke({
        "tool_name": tool_name,
        "tool_path": tool_path,
        "validation_level": "comprehensive"
    })
    
    validation_data = json.loads(validation_result)
    
    if validation_data["status"] == "success":
        result = validation_data["validation_result"]
        print(f"   Overall Status: {result['overall_status']}")
        print(f"   Security Score: {result['security_analysis']['score']}/100")
        
        # Show security analysis
        security = result["security_analysis"]
        if security["high_risk"]:
            print("   High Risk Issues:")
            for issue in security["high_risk"]:
                print(f"     - {issue}")
        
        if security["medium_risk"]:
            print("   Medium Risk Issues:")
            for issue in security["medium_risk"]:
                print(f"     - {issue}")
        
        # Show functionality tests
        print("   Functionality Tests:")
        for test_name, test_result in result["functionality_tests"].items():
            status_icon = "✅" if test_result["status"] == "passed" else "❌"
            print(f"     {status_icon} {test_name}: {test_result['details']}")
        
        # Show performance metrics
        perf = result["performance_metrics"]
        print(f"   Performance Metrics:")
        print(f"     File Size: {perf['file_size_bytes']} bytes")
        print(f"     Lines of Code: {perf['lines_of_code']}")
        print(f"     Complexity Score: {perf['complexity_score']:.1f}")
        
        # Show recommendations
        if result.get("recommendations"):
            print("   Recommendations:")
            for rec in result["recommendations"]:
                print(f"     - {rec}")
    
    print("\n3. Running security scan...")
    
    # Security scan
    security_result = security_scan_tool.invoke({
        "tool_path": tool_path,
        "scan_level": "strict"
    })
    
    security_data = json.loads(security_result)
    
    if security_data["status"] == "success":
        print(f"   Security Level: {security_data['security_level']}")
        print(f"   Security Score: {security_data['security_analysis']['score']}/100")
        
        if security_data["recommendations"]:
            print("   Security Recommendations:")
            for rec in security_data["recommendations"]:
                print(f"     - {rec}")
    
    print("\n4. Running custom sandbox test...")
    
    # Custom test
    test_code = f'''
import sys
sys.path.insert(0, '.')
from {tool_name} import {tool_name}

# Test basic functionality
try:
    result = {tool_name}.invoke({{"operation": "add", "a": 5, "b": 3}})
    print(f"Add test: {{result}}")
    
    result = {tool_name}.invoke({{"operation": "divide", "a": 10, "b": 2}})
    print(f"Divide test: {{result}}")
    
    # Test error handling
    result = {tool_name}.invoke({{"operation": "divide", "a": 10, "b": 0}})
    print(f"Error test: {{result}}")
    
    print("All tests completed successfully")
except Exception as e:
    print(f"Test failed: {{e}}")
'''
    
    test_result = run_sandbox_test.invoke({
        "tool_name": tool_name,
        "test_code": test_code,
        "timeout": 30
    })
    
    test_data = json.loads(test_result)
    
    if test_data["status"] == "success":
        result = test_data["test_result"]
        print(f"   Test Status: {result['status']}")
        print(f"   Execution Time: {result['execution_time']:.2f}s")
        if result["stdout"]:
            print("   Test Output:")
            for line in result["stdout"].split('\n')[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"     {line}")
    
    return validation_data, security_data, test_data


def demo_tool_documentation():
    """Demonstrate tool documentation capabilities."""
    print("\n" + "=" * 80)
    print("TOOL DOCUMENTATION DEMONSTRATION")
    print("=" * 80)
    
    tool_name = "sample_calculator"
    tool_path = "tools/sample_calculator.py"
    
    print("1. Generating comprehensive documentation...")
    
    # Generate documentation
    doc_result = generate_tool_documentation.invoke({
        "tool_name": tool_name,
        "tool_path": tool_path,
        "output_format": "markdown",
        "include_examples": True
    })
    
    doc_data = json.loads(doc_result)
    
    if doc_data["status"] == "success":
        print(f"   Documentation generated: {doc_data['documentation_path']}")
        print(f"   Format: {doc_data['format']}")
        
        summary = doc_data["analysis_summary"]
        print(f"   Analysis Summary:")
        print(f"     Function: {summary['function_name']}")
        print(f"     Parameters: {summary['parameter_count']}")
        print(f"     Has Docstring: {summary['has_docstring']}")
        print(f"     Has Examples: {summary['has_examples']}")
        print(f"     Complexity: {summary['complexity_score']}")
        
        # Show a snippet of the generated documentation
        try:
            with open(doc_data['documentation_path'], 'r') as f:
                content = f.read()
            
            print("\n   Documentation Preview:")
            lines = content.split('\n')
            for line in lines[:15]:  # Show first 15 lines
                print(f"     {line}")
            if len(lines) > 15:
                print("     ... (truncated)")
        except Exception as e:
            print(f"   Could not preview documentation: {e}")
    
    print("\n2. Updating documentation with additional information...")
    
    # Update documentation
    update_result = update_tool_documentation.invoke({
        "tool_name": tool_name,
        "additional_info": {
            "performance_notes": "Optimized for basic arithmetic operations",
            "version_history": "v1.0 - Initial release with basic operations",
            "known_limitations": "Does not support complex numbers or advanced operations"
        }
    })
    
    update_data = json.loads(update_result)
    
    if update_data["status"] == "success":
        print(f"   Documentation updated: {update_data['updated_file']}")
        print("   Added information:")
        for key, value in update_data["additional_info"].items():
            print(f"     - {key}: {value}")
    
    print("\n3. Validating documentation quality...")
    
    # Validate documentation
    validate_result = validate_tool_documentation.invoke({
        "tool_name": tool_name,
        "check_completeness": True
    })
    
    validate_data = json.loads(validate_result)
    
    if validate_data["status"] == "success":
        result = validate_data["validation_result"]
        print(f"   Overall Status: {validate_data['overall_status']}")
        print(f"   Overall Score: {validate_data['overall_score']:.1f}/100")
        print(f"   Completeness Score: {result['completeness_score']:.1f}/100")
        print(f"   Quality Score: {result['quality_score']:.1f}/100")
        
        if result["issues"]:
            print("   Issues Found:")
            for issue in result["issues"]:
                print(f"     - {issue}")
        
        if result["recommendations"]:
            print("   Recommendations:")
            for rec in result["recommendations"]:
                print(f"     - {rec}")
    
    print("\n4. Listing all tool documentation...")
    
    # List documentation
    list_result = list_tool_documentation.invoke({})
    list_data = json.loads(list_result)
    
    if list_data["status"] == "success":
        print(f"   Total Documentation Files: {list_data['total_count']}")
        for doc in list_data["documentation_list"]:
            print(f"     - {doc['tool_name']}: {doc['format']} ({doc['generated_at'][:10]})")
    
    return doc_data, update_data, validate_data


def demo_tool_lifecycle():
    """Demonstrate tool lifecycle management capabilities."""
    print("\n" + "=" * 80)
    print("TOOL LIFECYCLE DEMONSTRATION")
    print("=" * 80)
    
    tool_name = "sample_calculator"
    
    print("1. Analyzing tool usage patterns...")
    
    # Analyze usage
    usage_result = analyze_tool_usage.invoke({
        "days_threshold": 30,
        "include_stats": True
    })
    
    usage_data = json.loads(usage_result)
    
    if usage_data["status"] == "success":
        analysis = usage_data["analysis"]
        print(f"   Total Tools: {analysis['total_tools']}")
        print(f"   Analysis Period: {analysis['days_threshold']} days")
        
        print("   Status Distribution:")
        for status, count in analysis["by_status"].items():
            print(f"     - {status}: {count}")
        
        if analysis["unused_tools"]:
            print(f"   Unused Tools: {len(analysis['unused_tools'])}")
            for tool in analysis["unused_tools"][:3]:  # Show first 3
                print(f"     - {tool['name']}: {tool['usage_count']} uses")
        
        if analysis.get("statistics"):
            stats = analysis["statistics"]
            print(f"   Statistics:")
            print(f"     Total Usage: {stats['total_usage_count']}")
            print(f"     Average Usage: {stats['average_usage_per_tool']:.1f}")
            print(f"     Unused Percentage: {stats['unused_percentage']:.1f}%")
        
        if analysis["recommendations"]:
            print("   Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"     - {rec}")
    
    print("\n2. Updating tool status...")
    
    # Update status to experimental
    status_result = update_tool_status.invoke({
        "tool_name": tool_name,
        "new_status": "experimental",
        "reason": "Testing new features"
    })
    
    status_data = json.loads(status_result)
    
    if status_data["status"] == "success":
        print(f"   Status Updated: {status_data['old_status']} → {status_data['new_status']}")
        print(f"   Reason: {status_data['reason']}")
        print(f"   Updated At: {status_data['updated_at'][:19]}")
    
    print("\n3. Demonstrating deprecation process...")
    
    # Deprecate tool
    deprecate_result = deprecate_tool.invoke({
        "tool_name": tool_name,
        "reason": "Replaced by advanced_calculator with more features",
        "replacement_tool": "advanced_calculator",
        "deprecation_period_days": 30
    })
    
    deprecate_data = json.loads(deprecate_result)
    
    if deprecate_data["status"] == "success":
        print(f"   Tool Deprecated: {deprecate_data['tool_name']}")
        print(f"   Deprecation Date: {deprecate_data['deprecation_date'][:10]}")
        print(f"   Removal Date: {deprecate_data['removal_date'][:10]}")
        print(f"   Reason: {deprecate_data['reason']}")
        print(f"   Replacement: {deprecate_data['replacement_tool']}")
        print(f"   Backup Created: {deprecate_data['backup_path']}")
    
    print("\n4. Getting lifecycle status...")
    
    # Get lifecycle status
    lifecycle_result = get_tool_lifecycle_status.invoke({})
    lifecycle_data = json.loads(lifecycle_result)
    
    if lifecycle_data["status"] == "success":
        status = lifecycle_data["lifecycle_status"]
        print(f"   Total Tools: {status['total_tools']}")
        
        print("   Status Distribution:")
        for status_type, count in status["by_status"].items():
            print(f"     - {status_type}: {count}")
        
        if status["recent_changes"]:
            print(f"   Recent Changes: {len(status['recent_changes'])}")
            for change in status["recent_changes"][:3]:  # Show first 3
                print(f"     - {change['tool_name']}: {change['change_type']}")
        
        if status["upcoming_removals"]:
            print(f"   Upcoming Removals: {len(status['upcoming_removals'])}")
            for removal in status["upcoming_removals"]:
                print(f"     - {removal['tool_name']}: {removal['days_until_removal']} days")
        
        health = status["system_health"]
        print(f"   System Health:")
        print(f"     Active Tools: {health['active_percentage']:.1f}%")
        print(f"     Health Score: {health['health_score']:.1f}/100")
        print(f"     Needs Attention: {health['needs_attention']}")
    
    print("\n5. Demonstrating removal process (simulation)...")
    
    # Note: We won't actually remove the tool in the demo
    print("   Removal process would:")
    print("     1. Check if tool is deprecated and past removal date")
    print("     2. Create backup of all related files")
    print("     3. Remove tool file, tests, and documentation")
    print("     4. Update metadata to mark as obsolete")
    print("     5. Log removal action with timestamp")
    
    return usage_data, status_data, deprecate_data, lifecycle_data


def demo_integration():
    """Demonstrate how all three capabilities work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED ENHANCED TOOLMAKER DEMONSTRATION")
    print("=" * 80)
    
    print("1. Enhanced ToolMaker workflow simulation...")
    
    # Simulate the enhanced workflow
    workflow_steps = [
        "Planning Phase - Analyze tool requirements",
        "Design Phase - Plan tool architecture and validation",
        "Implementation Phase - Write tool code with best practices",
        "Validation Phase - Sandbox testing for safety and functionality",
        "Security Phase - Comprehensive security scanning",
        "Documentation Phase - Auto-generate comprehensive documentation",
        "Lifecycle Phase - Register tool in lifecycle management",
        "Quality Assurance Phase - Final validation and compliance check",
        "Deployment Phase - Deploy with monitoring and deprecation tracking"
    ]
    
    print("   Enhanced workflow steps:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"     {i}. {step}")
    
    print("\n2. Integration benefits...")
    
    integration_benefits = [
        "Sandboxed Validation - Tools tested in isolation before deployment",
        "Security Scanning - Comprehensive security analysis prevents vulnerabilities",
        "Auto Documentation - Complete documentation generated automatically",
        "Lifecycle Management - Full tool lifecycle tracking and management",
        "Deprecation Process - Structured deprecation with replacement guidance",
        "Usage Analytics - Data-driven decisions for tool maintenance",
        "Quality Assurance - Multi-layer validation ensures tool quality",
        "Backup Management - Automatic backups before any destructive operations"
    ]
    
    print("   Integration benefits:")
    for benefit in integration_benefits:
        print(f"     ✅ {benefit}")
    
    print("\n3. Enhanced capabilities summary:")
    
    capabilities = {
        "Tool Validation": {
            "Sandboxed Testing": "Isolated environment for safe testing",
            "Security Analysis": "AST-based security risk assessment",
            "Functionality Tests": "Automated import and structure validation",
            "Performance Metrics": "File size and complexity analysis",
            "Custom Testing": "User-defined test execution"
        },
        "Tool Documentation": {
            "Auto Generation": "Comprehensive docs from code analysis",
            "Multiple Formats": "Markdown, HTML, and JSON output",
            "Usage Examples": "Automatically generated examples",
            "Edge Cases": "Identified potential edge cases",
            "Quality Validation": "Documentation completeness scoring"
        },
        "Tool Lifecycle": {
            "Status Management": "Active, deprecated, obsolete, experimental, beta",
            "Deprecation Process": "Structured deprecation with timelines",
            "Usage Analytics": "Data-driven usage pattern analysis",
            "Backup Management": "Automatic backups before changes",
            "Removal Process": "Safe removal with validation checks"
        }
    }
    
    for category, features in capabilities.items():
        print(f"\n   {category}:")
        for feature, description in features.items():
            print(f"     🔧 {feature}: {description}")
    
    print("\n4. Quality assurance metrics:")
    
    qa_metrics = [
        "Security Score: 0-100 based on risk analysis",
        "Functionality Score: Based on test pass rate",
        "Documentation Score: Completeness and quality assessment",
        "Lifecycle Health: Overall tool ecosystem health",
        "Compliance Rating: Adherence to best practices",
        "Performance Rating: Efficiency and resource usage"
    ]
    
    for metric in qa_metrics:
        print(f"     📊 {metric}")
    
    return {
        "sandboxed_validation": True,
        "auto_documentation": True,
        "lifecycle_management": True,
        "security_scanning": True,
        "quality_assurance": True,
        "backup_management": True
    }


def main():
    """Run all demonstrations."""
    print("ENHANCED TOOLMAKER CAPABILITIES DEMONSTRATION")
    print("This demo shows the three key improvements:")
    print("1. Tool Validation - Sandboxed testing for functionality and safety")
    print("2. Tool Documentation - Auto-generation of comprehensive documentation")
    print("3. Tool Deprecation - Complete lifecycle management with deprecation process")
    print()
    
    try:
        # Run individual demonstrations
        validation_demo = demo_tool_validation()
        documentation_demo = demo_tool_documentation()
        lifecycle_demo = demo_tool_lifecycle()
        integration_demo = demo_integration()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("✅ Tool Validation: Sandboxed testing and security scanning")
        print("✅ Tool Documentation: Auto-generated comprehensive documentation")
        print("✅ Tool Lifecycle: Complete deprecation and management process")
        print("✅ Integration: All capabilities working together seamlessly")
        print()
        print("Enhanced ToolMaker is ready with:")
        print("  🔒 Sandboxed Validation - Safe testing in isolated environments")
        print("  📚 Auto Documentation - Comprehensive docs generated automatically")
        print("  🔄 Lifecycle Management - Complete tool lifecycle tracking")
        print("  🛡️ Security Scanning - AST-based security risk analysis")
        print("  📊 Usage Analytics - Data-driven tool maintenance decisions")
        print("  💾 Backup Management - Automatic backups before changes")
        print()
        print("The enhanced system provides robust, secure, and maintainable")
        print("tool development with comprehensive quality assurance and lifecycle management.")
        
        # Cleanup demo files
        print("\n" + "=" * 40)
        print("CLEANUP")
        print("=" * 40)
        print("Cleaning up demonstration files...")
        
        # Remove sample tool
        sample_tool = "tools/sample_calculator.py"
        if os.path.exists(sample_tool):
            os.remove(sample_tool)
            print(f"✅ Removed: {sample_tool}")
        
        # Clean up validation data
        cleanup_result = cleanup_validation_data.invoke({})
        cleanup_data = json.loads(cleanup_result)
        if cleanup_data["status"] == "success":
            print(f"✅ Cleaned up {cleanup_data['cleaned_items']} validation items")
        
        # Clean up lifecycle data
        lifecycle_cleanup = cleanup_tool_lifecycle.invoke({})
        lifecycle_cleanup_data = json.loads(lifecycle_cleanup)
        if lifecycle_cleanup_data["status"] == "success":
            results = lifecycle_cleanup_data["cleanup_results"]
            total_cleaned = sum(results.values())
            print(f"✅ Cleaned up {total_cleaned} lifecycle items")
        
        print("Demo cleanup completed!")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the enhanced capabilities structure.")
        print("Full integration requires the complete enhanced ToolMaker system.")


if __name__ == "__main__":
    main()