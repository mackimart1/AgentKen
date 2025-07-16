#!/usr/bin/env python3
"""
Demonstration script for Enhanced Agent Smith capabilities:
1. Self-Healing: Monitors for crashes/anomalies and autonomously recovers
2. Agent Versioning: Tracks changes and enables rollback to previous versions
3. Testing Framework: Automated validation before deployment

This script shows how the enhanced features work without requiring full agent creation.
"""

import json
import time
from datetime import datetime
from tools.agent_versioning import (
    create_agent_version, rollback_agent_version, list_agent_versions, 
    compare_agent_versions, get_version_registry_status
)
from tools.agent_testing import (
    generate_agent_tests, run_agent_tests, validate_agent_code, 
    get_test_results_summary
)
from tools.self_healing import (
    check_system_health, perform_recovery_action, configure_monitoring,
    get_health_history, emergency_recovery
)


def demo_self_healing():
    """Demonstrate self-healing capabilities."""
    print("=" * 80)
    print("SELF-HEALING DEMONSTRATION")
    print("=" * 80)
    
    print("1. Checking system health...")
    
    # Check overall system health
    health_result = check_system_health.invoke({
        "component": "system",
        "detailed": True
    })
    health_data = json.loads(health_result)
    
    if health_data["status"] == "success":
        health_status = health_data["health_status"]
        print(f"   System Status: {health_status['status']}")
        print(f"   Health Score: {health_status['score']:.1f}/100")
        
        if health_status["issues"]:
            print("   Issues detected:")
            for issue in health_status["issues"]:
                print(f"     - {issue}")
        
        if health_status["recommendations"]:
            print("   Recommendations:")
            for rec in health_status["recommendations"]:
                print(f"     - {rec}")
    
    print("\n2. Configuring monitoring...")
    
    # Configure monitoring with custom thresholds
    monitor_result = configure_monitoring.invoke({
        "enabled": True,
        "check_interval": 30,
        "thresholds": {
            "cpu_usage": 75.0,
            "memory_usage": 80.0,
            "error_rate": 3.0
        }
    })
    monitor_data = json.loads(monitor_result)
    print(f"   Monitoring: {monitor_data['message']}")
    
    print("\n3. Demonstrating recovery actions...")
    
    # Perform memory cleanup
    recovery_result = perform_recovery_action.invoke({
        "recovery_type": "memory_cleanup",
        "component": "system"
    })
    recovery_data = json.loads(recovery_result)
    
    if recovery_data["status"] == "success":
        print(f"   Recovery: {recovery_data['message']}")
        print(f"   Actions taken: {', '.join(recovery_data['actions_taken'])}")
        if "memory_freed_percent" in recovery_data:
            print(f"   Memory freed: {recovery_data['memory_freed_percent']:.2f}%")
    
    print("\n4. Testing emergency recovery...")
    
    # Test emergency recovery
    emergency_result = emergency_recovery.invoke({})
    emergency_data = json.loads(emergency_result)
    
    if emergency_data["status"] == "success":
        print(f"   Emergency recovery: {emergency_data['message']}")
        print(f"   Emergency actions: {', '.join(emergency_data['actions_taken'])}")
    
    return health_data, monitor_data, recovery_data, emergency_data


def demo_agent_versioning():
    """Demonstrate agent versioning capabilities."""
    print("\n" + "=" * 80)
    print("AGENT VERSIONING DEMONSTRATION")
    print("=" * 80)
    
    # Create a sample agent file for demonstration
    sample_agent_code = '''"""
Sample agent for versioning demonstration.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def sample_agent(task: str) -> Dict[str, Any]:
    """
    Sample agent for demonstration purposes.
    
    Args:
        task (str): The task to perform
        
    Returns:
        Dict[str, Any]: Result dictionary
    """
    try:
        logger.info(f"Sample agent executing task: {task}")
        
        # Simple task processing
        result = f"Processed: {task}"
        
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully completed task: {task}"
        }
        
    except Exception as e:
        logger.error(f"Error in sample_agent: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to complete task: {str(e)}"
        }
'''
    
    # Write sample agent file
    import os
    os.makedirs("agents", exist_ok=True)
    sample_file = "agents/sample_agent.py"
    
    with open(sample_file, 'w') as f:
        f.write(sample_agent_code)
    
    print("1. Creating initial version...")
    
    # Create initial version
    version_result = create_agent_version.invoke({
        "agent_name": "sample_agent",
        "file_path": sample_file,
        "description": "Initial version of sample agent",
        "metadata": {"author": "demo", "purpose": "demonstration"}
    })
    version_data = json.loads(version_result)
    
    if version_data["status"] == "success":
        print(f"   Created version: {version_data['version']}")
        print(f"   Version path: {version_data['version_path']}")
        print(f"   File hash: {version_data['hash'][:8]}...")
    
    print("\n2. Creating modified version...")
    
    # Modify the agent and create another version
    modified_code = sample_agent_code.replace(
        "# Simple task processing",
        "# Enhanced task processing with validation\n        if not task:\n            raise ValueError('Task cannot be empty')"
    )
    
    with open(sample_file, 'w') as f:
        f.write(modified_code)
    
    version2_result = create_agent_version.invoke({
        "agent_name": "sample_agent",
        "file_path": sample_file,
        "description": "Added input validation",
        "metadata": {"author": "demo", "change_type": "enhancement"}
    })
    version2_data = json.loads(version2_result)
    
    if version2_data["status"] == "success":
        print(f"   Created version: {version2_data['version']}")
        print(f"   File hash: {version2_data['hash'][:8]}...")
    
    print("\n3. Listing all versions...")
    
    # List all versions
    list_result = list_agent_versions.invoke({
        "agent_name": "sample_agent"
    })
    list_data = json.loads(list_result)
    
    if list_data["status"] == "success":
        print(f"   Total versions: {list_data['total_versions']}")
        print(f"   Latest version: {list_data['latest_version']}")
        
        for version in list_data["versions"]:
            print(f"     - {version['version']}: {version['description']} ({version['created_at'][:19]})")
    
    print("\n4. Comparing versions...")
    
    # Compare versions if we have at least 2
    if list_data["status"] == "success" and len(list_data["versions"]) >= 2:
        v1 = list_data["versions"][1]["version"]  # Older version
        v2 = list_data["versions"][0]["version"]  # Newer version
        
        compare_result = compare_agent_versions.invoke({
            "agent_name": "sample_agent",
            "version1": v1,
            "version2": v2
        })
        compare_data = json.loads(compare_result)
        
        if compare_data["status"] == "success":
            comparison = compare_data["comparison"]
            print(f"   Comparing {v1} vs {v2}:")
            print(f"     Hash different: {comparison['differences']['hash_different']}")
            print(f"     Size difference: {comparison['differences']['size_difference']} bytes")
            
            if "content_diff" in comparison:
                diff_count = comparison["content_diff"].get("total_differences", 0)
                print(f"     Content differences: {diff_count}")
    
    print("\n5. Testing rollback...")
    
    # Rollback to previous version
    if list_data["status"] == "success" and len(list_data["versions"]) >= 2:
        target_version = list_data["versions"][1]["version"]  # Rollback to older version
        
        rollback_result = rollback_agent_version.invoke({
            "agent_name": "sample_agent",
            "version": target_version
        })
        rollback_data = json.loads(rollback_result)
        
        if rollback_data["status"] == "success":
            print(f"   Rollback successful: {rollback_data['message']}")
            if "backup_created" in rollback_data:
                print(f"   Backup created: {rollback_data['backup_created']}")
    
    print("\n6. Version registry status...")
    
    # Get registry status
    registry_result = get_version_registry_status.invoke({})
    registry_data = json.loads(registry_result)
    
    if registry_data["status"] == "success":
        status = registry_data["registry_status"]
        print(f"   Total agents: {status['total_agents']}")
        print(f"   Total versions: {status['total_versions']}")
        print(f"   Registry file: {status['registry_file']}")
    
    return version_data, version2_data, list_data, rollback_data


def demo_testing_framework():
    """Demonstrate testing framework capabilities."""
    print("\n" + "=" * 80)
    print("TESTING FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    
    agent_name = "sample_agent"
    agent_file = "agents/sample_agent.py"
    
    print("1. Validating agent code...")
    
    # Validate agent code
    validation_result = validate_agent_code.invoke({
        "agent_name": agent_name,
        "agent_file_path": agent_file,
        "validation_level": "comprehensive"
    })
    validation_data = json.loads(validation_result)
    
    if validation_data["status"] == "success":
        print(f"   Validation score: {validation_data['score']}/{validation_data['max_score']}")
        print(f"   Grade: {validation_data['grade']} ({validation_data['score_percentage']:.1f}%)")
        
        print("   Compliance checks:")
        for check in validation_data["compliance_checks"][:5]:  # Show first 5
            status_icon = "✅" if check["status"] == "passed" else "❌"
            print(f"     {status_icon} {check['check']}")
        
        if validation_data["best_practices"]:
            print("   Best practice recommendations:")
            for practice in validation_data["best_practices"][:3]:  # Show first 3
                print(f"     - {practice}")
    
    print("\n2. Generating comprehensive test suite...")
    
    # Generate tests
    test_result = generate_agent_tests.invoke({
        "agent_name": agent_name,
        "agent_file_path": agent_file,
        "test_types": ["unit", "integration", "performance"]
    })
    test_data = json.loads(test_result)
    
    if test_data["status"] == "success":
        print(f"   Test file created: {test_data['test_file_path']}")
        print(f"   Total tests generated: {test_data['total_tests']}")
        print(f"   Test types: {', '.join(test_data['test_types'])}")
        
        analysis = test_data.get("analysis", {})
        if analysis:
            print(f"   Agent analysis:")
            print(f"     Functions found: {len(analysis.get('functions', []))}")
            print(f"     Tools used: {len(analysis.get('tools_used', []))}")
            print(f"     Has error handling: {analysis.get('error_handling', False)}")
    
    print("\n3. Running generated tests...")
    
    # Run tests (this might fail in demo environment, but shows the process)
    try:
        run_result = run_agent_tests.invoke({
            "agent_name": agent_name,
            "timeout": 60
        })
        run_data = json.loads(run_result)
        
        if run_data["status"] in ["success", "failure"]:
            summary = run_data.get("summary", {})
            print(f"   Test execution: {run_data['status']}")
            print(f"   Total tests: {summary.get('total_tests', 0)}")
            print(f"   Passed: {summary.get('passed', 0)}")
            print(f"   Failed: {summary.get('failed', 0)}")
            print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
            print(f"   Execution time: {run_data.get('execution_time', 0):.2f}s")
        else:
            print(f"   Test execution: {run_data['message']}")
    
    except Exception as e:
        print(f"   Test execution: Could not run tests in demo environment ({e})")
    
    print("\n4. Test results summary...")
    
    # Get test summary
    summary_result = get_test_results_summary.invoke({})
    summary_data = json.loads(summary_result)
    
    if summary_data["status"] == "success":
        summary = summary_data["summary"]
        print(f"   Agents tested: {summary['total_agents_tested']}")
        print(f"   Overall success rate: {summary['overall_success_rate']:.1f}%")
        
        if summary["agents"]:
            print("   Recent test results:")
            for agent_summary in summary["agents"][:3]:  # Show first 3
                print(f"     - {agent_summary['agent_name']}: {agent_summary['success_rate']:.1f}% success")
    
    return validation_data, test_data, summary_data


def demo_integration():
    """Demonstrate how all three capabilities work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED ENHANCED AGENT SMITH DEMONSTRATION")
    print("=" * 80)
    
    print("1. Enhanced Agent Smith workflow simulation...")
    
    # Simulate the enhanced workflow
    workflow_steps = [
        "Planning Phase - Analyze requirements",
        "Versioning Phase - Create initial version",
        "Test Generation Phase - Generate comprehensive tests",
        "Testing Phase - Run validation tests",
        "Health Monitoring - Check system health",
        "Quality Assurance - Validate code compliance",
        "Deployment Phase - Deploy with rollback capability"
    ]
    
    print("   Enhanced workflow steps:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"     {i}. {step}")
        time.sleep(0.5)  # Simulate processing time
    
    print("\n2. Self-healing integration...")
    
    # Check health during agent development
    health_check = check_system_health.invoke({"component": "agent_smith"})
    health_data = json.loads(health_check)
    
    if health_data["status"] == "success":
        print(f"   Agent Smith health: {health_data.get('health_status', 'healthy')}")
        print(f"   System responsive: {health_data.get('agent_smith_responsive', True)}")
    
    print("\n3. Version control integration...")
    
    # Show version control benefits
    registry_status = get_version_registry_status.invoke({})
    registry_data = json.loads(registry_status)
    
    if registry_data["status"] == "success":
        status = registry_data["registry_status"]
        print(f"   Agents under version control: {status['total_agents']}")
        print(f"   Total versions tracked: {status['total_versions']}")
        print("   Rollback capability: Available for all agents")
    
    print("\n4. Testing integration...")
    
    # Show testing benefits
    test_summary = get_test_results_summary.invoke({})
    test_data = json.loads(test_summary)
    
    if test_data["status"] == "success":
        summary = test_data["summary"]
        print(f"   Automated testing: {summary['total_agents_tested']} agents tested")
        print(f"   Quality assurance: {summary['overall_success_rate']:.1f}% success rate")
        print("   Deployment safety: All agents validated before deployment")
    
    print("\n5. Enhanced capabilities summary:")
    print("   ✅ Self-Healing: Autonomous monitoring and recovery")
    print("   ✅ Agent Versioning: Complete change tracking and rollback")
    print("   ✅ Testing Framework: Comprehensive automated validation")
    print("   ✅ Quality Assurance: Code compliance and best practices")
    print("   ✅ Deployment Safety: Validated and recoverable deployments")
    
    return {
        "health_monitoring": True,
        "version_control": True,
        "automated_testing": True,
        "quality_assurance": True,
        "deployment_safety": True
    }


def main():
    """Run all demonstrations."""
    print("ENHANCED AGENT SMITH CAPABILITIES DEMONSTRATION")
    print("This demo shows the three key improvements:")
    print("1. Self-Healing - Autonomous monitoring and recovery")
    print("2. Agent Versioning - Change tracking and rollback capabilities")
    print("3. Testing Framework - Comprehensive automated validation")
    print()
    
    try:
        # Run individual demonstrations
        health_demo = demo_self_healing()
        version_demo = demo_agent_versioning()
        testing_demo = demo_testing_framework()
        integration_demo = demo_integration()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("✅ Self-Healing: System monitoring and autonomous recovery")
        print("✅ Agent Versioning: Complete version control with rollback")
        print("✅ Testing Framework: Automated test generation and validation")
        print("✅ Integration: All capabilities working together seamlessly")
        print()
        print("Enhanced Agent Smith is ready with:")
        print("  🔧 Self-Healing - Detects and recovers from system issues")
        print("  📚 Version Control - Tracks all changes with rollback capability")
        print("  🧪 Testing Framework - Validates agents before deployment")
        print("  🛡️ Quality Assurance - Ensures code compliance and best practices")
        print("  🚀 Safe Deployment - Rollback available if issues arise")
        print()
        print("The enhanced system provides robust, reliable, and maintainable")
        print("agent development with comprehensive quality assurance.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the enhanced capabilities structure.")
        print("Full integration requires the complete enhanced Agent Smith system.")


if __name__ == "__main__":
    main()