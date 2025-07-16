#!/usr/bin/env python3
"""
Demonstration script for Error Handling & Recovery System capabilities:
1. Centralized Error-Handling Agent: Logs exceptions, analyzes failure patterns, suggests fixes
2. Retry Logic with Exponential Backoff: Handles transient failures gracefully

This script shows how the enhanced error handling system works with comprehensive
error analysis, pattern recognition, and intelligent retry mechanisms.
"""

import json
import time
import random
from datetime import datetime

# Import the error handling system tools
from tools.error_handling_system import (
    log_error_centralized, resolve_error_centralized,
    get_error_analysis, get_errors_by_pattern,
    get_retry_statistics, get_error_categories_info,
    with_retry, RetryConfig, RetryStrategy
)


def demo_centralized_error_handling():
    """Demonstrate centralized error handling capabilities."""
    print("=" * 80)
    print("CENTRALIZED ERROR HANDLING DEMONSTRATION")
    print("=" * 80)
    
    print("1. Getting error categories and system information...")
    
    # Get error categories info
    categories_result = get_error_categories_info.invoke({})
    categories_data = json.loads(categories_result)
    
    if categories_data["status"] == "success":
        categories = categories_data["categories_info"]["error_categories"]
        severities = categories_data["categories_info"]["severity_levels"]
        strategies = categories_data["categories_info"]["retry_strategies"]
        
        print(f"   📋 Error Categories: {len(categories)} available")
        print(f"   🚨 Severity Levels: {len(severities)} levels")
        print(f"   🔄 Retry Strategies: {len(strategies)} strategies")
        
        print("\n   Key Error Categories:")
        for category, info in list(categories.items())[:5]:
            print(f"     - {category}: {info['description']}")
            print(f"       Typical fixes: {len(info['typical_fixes'])} suggestions")
    
    print("\n2. Simulating various error scenarios...")
    
    # Simulate different types of errors
    error_scenarios = [
        {
            "exception_type": "ConnectionError",
            "error_message": "Connection timeout while connecting to database",
            "agent_id": "database_agent",
            "tool_name": "database_connector",
            "function_name": "connect_to_db",
            "parameters": {"host": "db.example.com", "port": 5432, "timeout": 30},
            "stack_trace": "Traceback (most recent call last):\n  File 'db.py', line 42, in connect\n    ConnectionError: timeout"
        },
        {
            "exception_type": "AuthenticationError",
            "error_message": "Invalid API key provided",
            "agent_id": "api_client",
            "tool_name": "external_api",
            "function_name": "authenticate",
            "parameters": {"api_key": "***masked***", "endpoint": "/auth"},
            "stack_trace": "Traceback (most recent call last):\n  File 'api.py', line 15, in auth\n    AuthenticationError: 401 Unauthorized"
        },
        {
            "exception_type": "PermissionError",
            "error_message": "Access denied to file /etc/config.conf",
            "agent_id": "file_manager",
            "tool_name": "file_operations",
            "function_name": "read_config",
            "parameters": {"file_path": "/etc/config.conf", "mode": "r"},
            "stack_trace": "Traceback (most recent call last):\n  File 'files.py', line 28, in read\n    PermissionError: [Errno 13] Permission denied"
        },
        {
            "exception_type": "ValidationError",
            "error_message": "Invalid input format: expected JSON, got XML",
            "agent_id": "data_processor",
            "tool_name": "data_validator",
            "function_name": "validate_input",
            "parameters": {"data": "<xml>...</xml>", "format": "json"},
            "stack_trace": "Traceback (most recent call last):\n  File 'validator.py', line 55, in validate\n    ValidationError: Invalid format"
        },
        {
            "exception_type": "ResourceExhaustedError",
            "error_message": "Out of memory: cannot allocate 2GB",
            "agent_id": "ml_processor",
            "tool_name": "model_trainer",
            "function_name": "train_model",
            "parameters": {"model_size": "large", "batch_size": 1024},
            "stack_trace": "Traceback (most recent call last):\n  File 'ml.py', line 120, in train\n    MemoryError: Cannot allocate memory"
        }
    ]
    
    logged_errors = []
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['exception_type']}")
        
        # Log the error
        result = log_error_centralized.invoke(scenario)
        result_data = json.loads(result)
        
        if result_data["status"] == "success":
            error_record = result_data["error_record"]
            error_id = error_record["error_id"]
            logged_errors.append(error_id)
            
            print(f"     ✅ Error logged: {error_id}")
            print(f"     📊 Category: {error_record['error_category']}")
            print(f"     🚨 Severity: {error_record['severity']}")
            print(f"     💡 Suggestions: {len(error_record['suggested_fixes'])} fixes")
            
            # Show first suggestion
            if error_record['suggested_fixes']:
                print(f"     🔧 Top suggestion: {error_record['suggested_fixes'][0]}")
        else:
            print(f"     ❌ Failed to log error: {result_data['message']}")
    
    print(f"\n3. Analyzing error patterns and trends...")
    
    # Get comprehensive error analysis
    analysis_result = get_error_analysis.invoke({
        "hours_back": 1,  # Look at recent errors
        "include_suggestions": True
    })
    
    analysis_data = json.loads(analysis_result)
    if analysis_data["status"] == "success":
        analysis = analysis_data["analysis"]
        stats = analysis["statistics"]
        
        print(f"   📊 Total errors in last hour: {stats['total_errors']}")
        print(f"   ✅ Resolved errors: {stats['resolved_errors']}")
        print(f"   📈 Resolution rate: {stats['resolution_rate']:.1f}%")
        
        if stats.get("category_distribution"):
            print(f"\n   📋 Error distribution by category:")
            for category, count in stats["category_distribution"].items():
                print(f"     - {category}: {count} errors")
        
        if stats.get("top_error_types"):
            print(f"\n   🔝 Top error types:")
            for error_info in stats["top_error_types"][:3]:
                print(f"     - {error_info['error_type']}: {error_info['count']} occurrences")
        
        if analysis.get("suggested_fixes"):
            print(f"\n   💡 System-wide fix suggestions:")
            for suggestion in analysis["suggested_fixes"][:3]:
                print(f"     - {suggestion}")
    
    print(f"\n4. Demonstrating error resolution tracking...")
    
    # Resolve some errors
    if logged_errors:
        error_to_resolve = logged_errors[0]
        resolution_result = resolve_error_centralized.invoke({
            "error_id": error_to_resolve,
            "resolution_notes": "Fixed by updating connection timeout configuration and implementing connection pooling"
        })
        
        resolution_data = json.loads(resolution_result)
        if resolution_data["status"] == "success":
            print(f"   ✅ Error {error_to_resolve[:8]}... marked as resolved")
            print(f"   📝 Resolution tracked for future reference")
        else:
            print(f"   ❌ Failed to resolve error: {resolution_data['message']}")
    
    return logged_errors


def demo_retry_logic_with_backoff():
    """Demonstrate retry logic with exponential backoff."""
    print("\n" + "=" * 80)
    print("RETRY LOGIC WITH EXPONENTIAL BACKOFF DEMONSTRATION")
    print("=" * 80)
    
    print("1. Testing retry decorators with different strategies...")
    
    # Define test functions with different retry configurations
    
    @with_retry(RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exponential_base=2.0
    ))
    def flaky_network_call(agent_id="network_agent", tool_name="network_tool"):
        """Simulates a flaky network call that sometimes fails."""
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Network timeout")
        return "Network call successful"
    
    @with_retry(RetryConfig(
        max_attempts=4,
        base_delay=0.2,
        strategy=RetryStrategy.LINEAR_BACKOFF
    ))
    def database_operation(agent_id="db_agent", tool_name="db_tool"):
        """Simulates a database operation that might fail due to locks."""
        if random.random() < 0.5:  # 50% chance of failure
            raise TimeoutError("Database lock timeout")
        return "Database operation completed"
    
    @with_retry(RetryConfig(
        max_attempts=2,
        strategy=RetryStrategy.NO_RETRY
    ))
    def validation_check(agent_id="validator_agent", tool_name="validator_tool"):
        """Simulates a validation that should not be retried."""
        if random.random() < 0.8:  # 80% chance of failure
            raise ValueError("Invalid input format")
        return "Validation passed"
    
    # Test the retry functions
    test_functions = [
        ("Exponential Backoff Network Call", flaky_network_call),
        ("Linear Backoff Database Operation", database_operation),
        ("No Retry Validation", validation_check)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n   Testing: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func()
            execution_time = time.time() - start_time
            print(f"     ✅ Success: {result}")
            print(f"     ⏱️  Execution time: {execution_time:.2f}s")
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"     ❌ Failed after retries: {str(e)}")
            print(f"     ⏱️  Total time: {execution_time:.2f}s")
    
    print(f"\n2. Analyzing retry statistics...")
    
    # Get retry statistics
    retry_stats_result = get_retry_statistics.invoke({})
    retry_stats_data = json.loads(retry_stats_result)
    
    if retry_stats_data["status"] == "success":
        stats = retry_stats_data["retry_statistics"]
        success_rates = retry_stats_data["success_rates"]
        
        print(f"   📊 Retry Statistics:")
        if stats:
            for key, value in list(stats.items())[:5]:  # Show first 5 stats
                print(f"     - {key}: {value}")
        else:
            print(f"     - No retry statistics available yet")
        
        if success_rates:
            print(f"\n   📈 Success Rates:")
            for key, rate in list(success_rates.items())[:3]:
                print(f"     - {key}: {rate:.1f}%")
    
    print(f"\n3. Testing different retry strategies...")
    
    # Demonstrate different retry strategies
    strategies_demo = [
        {
            "name": "Exponential Backoff",
            "config": RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                exponential_base=2.0
            ),
            "description": "Delays: 0.1s, 0.2s, 0.4s"
        },
        {
            "name": "Linear Backoff",
            "config": RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                strategy=RetryStrategy.LINEAR_BACKOFF
            ),
            "description": "Delays: 0.1s, 0.2s, 0.3s"
        },
        {
            "name": "Fixed Delay",
            "config": RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                strategy=RetryStrategy.FIXED_DELAY
            ),
            "description": "Delays: 0.1s, 0.1s, 0.1s"
        }
    ]
    
    for strategy_info in strategies_demo:
        print(f"\n   Strategy: {strategy_info['name']}")
        print(f"   Description: {strategy_info['description']}")
        
        @with_retry(strategy_info["config"])
        def test_strategy_function(agent_id="test_agent", tool_name="test_tool"):
            # Always fail to show retry pattern
            raise ConnectionError("Simulated failure for retry demo")
        
        start_time = time.time()
        try:
            test_strategy_function()
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"     ⏱️  Total retry time: {execution_time:.2f}s")
            print(f"     🔄 Retry pattern demonstrated")
    
    return retry_stats_data


def demo_pattern_analysis():
    """Demonstrate error pattern analysis and suggestions."""
    print("\n" + "=" * 80)
    print("ERROR PATTERN ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    print("1. Querying errors by specific patterns...")
    
    # Query errors by different patterns
    pattern_queries = [
        {
            "name": "Network-related errors",
            "params": {"error_category": "network", "hours_back": 24}
        },
        {
            "name": "Authentication failures",
            "params": {"error_category": "authentication", "hours_back": 24}
        },
        {
            "name": "Database agent errors",
            "params": {"agent_id": "database_agent", "hours_back": 24}
        },
        {
            "name": "Validation tool errors",
            "params": {"tool_name": "data_validator", "hours_back": 24}
        }
    ]
    
    for query_info in pattern_queries:
        print(f"\n   Querying: {query_info['name']}")
        
        result = get_errors_by_pattern.invoke(query_info["params"])
        result_data = json.loads(result)
        
        if result_data["status"] == "success":
            errors = result_data["errors"]
            count = result_data["count"]
            
            print(f"     📊 Found {count} matching errors")
            
            if errors:
                # Show details of first error
                first_error = errors[0]
                print(f"     🔍 Sample error:")
                print(f"       Type: {first_error['error_type']}")
                print(f"       Message: {first_error['error_message'][:60]}...")
                print(f"       Agent: {first_error['context']['agent_id']}")
                print(f"       Tool: {first_error['context']['tool_name']}")
                
                if first_error.get('suggested_fixes'):
                    print(f"       💡 Suggestions: {len(first_error['suggested_fixes'])} available")
        else:
            print(f"     ❌ Query failed: {result_data['message']}")
    
    print(f"\n2. Analyzing failure patterns and trends...")
    
    # Get comprehensive analysis
    analysis_result = get_error_analysis.invoke({
        "hours_back": 24,
        "include_suggestions": True
    })
    
    analysis_data = json.loads(analysis_result)
    if analysis_data["status"] == "success":
        analysis = analysis_data["analysis"]
        
        print(f"   📈 Pattern Analysis Results:")
        
        # Show critical errors
        if analysis.get("critical_errors"):
            print(f"     🚨 Critical errors: {len(analysis['critical_errors'])}")
        
        # Show unresolved high severity errors
        if analysis.get("unresolved_high_severity"):
            print(f"     ⚠️  Unresolved high severity: {len(analysis['unresolved_high_severity'])}")
        
        # Show retry statistics
        if analysis.get("retry_statistics"):
            retry_stats = analysis["retry_statistics"]
            total_attempts = sum(v for k, v in retry_stats.items() if "_attempts" in k)
            if total_attempts > 0:
                print(f"     🔄 Total retry attempts: {total_attempts}")
        
        # Show suggested fixes
        if analysis.get("suggested_fixes"):
            print(f"\n   💡 Top system-wide recommendations:")
            for i, suggestion in enumerate(analysis["suggested_fixes"][:3], 1):
                print(f"     {i}. {suggestion}")
    
    print(f"\n3. Demonstrating intelligent error categorization...")
    
    # Show how different error messages are categorized
    categorization_examples = [
        "Connection timeout after 30 seconds",
        "Invalid API key provided",
        "Permission denied to access file",
        "Out of memory error",
        "Invalid JSON format in request",
        "Database connection refused",
        "Module 'requests' not found",
        "Configuration file missing"
    ]
    
    print(f"   🧠 Automatic Error Categorization:")
    for example in categorization_examples:
        # This would normally be done internally by the error handler
        print(f"     '{example[:40]}...'")
        
        # Simulate categorization (in real system this happens automatically)
        if "timeout" in example.lower() or "connection" in example.lower():
            category = "network/timeout"
        elif "api key" in example.lower() or "unauthorized" in example.lower():
            category = "authentication"
        elif "permission" in example.lower() or "access" in example.lower():
            category = "permission"
        elif "memory" in example.lower():
            category = "resource"
        elif "format" in example.lower() or "json" in example.lower():
            category = "validation"
        elif "not found" in example.lower() and "module" in example.lower():
            category = "dependency"
        elif "config" in example.lower():
            category = "configuration"
        else:
            category = "unknown"
        
        print(f"       → Categorized as: {category}")
    
    return analysis_data


def demo_integration_benefits():
    """Demonstrate how centralized error handling and retry logic work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED ERROR HANDLING & RECOVERY BENEFITS")
    print("=" * 80)
    
    print("1. Comprehensive error management workflow...")
    
    workflow_steps = [
        "Error Occurrence - Exception happens in agent or tool",
        "Retry Logic - Automatic retry with exponential backoff",
        "Error Logging - Centralized logging with context",
        "Pattern Analysis - Automatic categorization and analysis",
        "Suggestion Generation - AI-powered fix recommendations",
        "Resolution Tracking - Mark errors as resolved with notes",
        "Trend Analysis - Identify patterns and recurring issues",
        "Proactive Prevention - Use insights to prevent future errors"
    ]
    
    print("   🔄 Error Management Workflow:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"     {i}. {step}")
    
    print("\n2. System reliability improvements...")
    
    reliability_benefits = [
        "Automatic Recovery - Retry logic handles transient failures",
        "Comprehensive Logging - All errors tracked with full context",
        "Pattern Recognition - Identify recurring issues automatically",
        "Actionable Insights - AI-generated fix suggestions",
        "Trend Analysis - Understand error patterns over time",
        "Resolution Tracking - Monitor fix effectiveness",
        "Proactive Prevention - Use data to prevent future issues",
        "System Health Monitoring - Overall error rate tracking"
    ]
    
    print("   🛡️  Reliability Benefits:")
    for benefit in reliability_benefits:
        print(f"     ✅ {benefit}")
    
    print("\n3. Enterprise-grade error handling features...")
    
    enterprise_features = {
        "Centralized Error Agent": {
            "Error Classification": "Automatic categorization by type and severity",
            "Pattern Analysis": "ML-powered pattern recognition and clustering",
            "Fix Suggestions": "Context-aware actionable recommendations",
            "Trend Monitoring": "Long-term error trend analysis and reporting",
            "Resolution Tracking": "Complete lifecycle management of errors"
        },
        "Retry Logic System": {
            "Exponential Backoff": "Intelligent delay calculation with jitter",
            "Strategy Selection": "Multiple retry strategies for different scenarios",
            "Exception Filtering": "Smart retry decisions based on error type",
            "Performance Monitoring": "Retry effectiveness tracking and optimization",
            "Resource Protection": "Prevent system overload during retries"
        }
    }
    
    for category, features in enterprise_features.items():
        print(f"\n   {category}:")
        for feature, description in features.items():
            print(f"     🔧 {feature}: {description}")
    
    print("\n4. Real-world impact scenarios...")
    
    impact_scenarios = [
        {
            "scenario": "Database Connection Issues",
            "before": "Manual investigation, service downtime, user impact",
            "after": "Automatic retry, error logging, pattern analysis, proactive fixes"
        },
        {
            "scenario": "API Authentication Failures",
            "before": "Silent failures, difficult debugging, repeated issues",
            "after": "Immediate detection, categorization, fix suggestions, resolution tracking"
        },
        {
            "scenario": "Resource Exhaustion",
            "before": "System crashes, data loss, long recovery times",
            "after": "Early detection, graceful degradation, automated recovery, capacity planning"
        },
        {
            "scenario": "Configuration Errors",
            "before": "Trial and error debugging, inconsistent fixes",
            "after": "Automatic detection, specific suggestions, knowledge base building"
        }
    ]
    
    for scenario in impact_scenarios:
        print(f"\n   📊 {scenario['scenario']}:")
        print(f"     Before: {scenario['before']}")
        print(f"     After: {scenario['after']}")
    
    print("\n5. Key metrics and monitoring...")
    
    key_metrics = [
        "Error Rate - Total errors per hour/day",
        "Resolution Rate - Percentage of errors resolved",
        "Retry Success Rate - Effectiveness of retry logic",
        "Mean Time to Resolution - Average time to fix errors",
        "Error Category Distribution - Types of errors occurring",
        "Agent/Tool Error Rates - Component-specific reliability",
        "Pattern Recognition Accuracy - Quality of categorization",
        "Fix Suggestion Effectiveness - Success rate of recommendations"
    ]
    
    print("   📈 Monitoring Metrics:")
    for metric in key_metrics:
        print(f"     📊 {metric}")
    
    return enterprise_features


def main():
    """Run all demonstrations."""
    print("ERROR HANDLING & RECOVERY SYSTEM DEMONSTRATION")
    print("This demo shows the two key improvements:")
    print("1. Centralized Error-Handling Agent - Logs, analyzes, and suggests fixes")
    print("2. Retry Logic with Exponential Backoff - Handles transient failures gracefully")
    print()
    
    try:
        # Run individual demonstrations
        error_handling_demo = demo_centralized_error_handling()
        retry_logic_demo = demo_retry_logic_with_backoff()
        pattern_analysis_demo = demo_pattern_analysis()
        integration_demo = demo_integration_benefits()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("✅ Centralized Error Handling: Comprehensive logging and analysis")
        print("✅ Retry Logic with Backoff: Intelligent failure recovery")
        print("✅ Pattern Analysis: AI-powered error categorization and insights")
        print("✅ Integration Benefits: Complete error management workflow")
        print()
        print("Error Handling & Recovery System is ready with:")
        print("  🎯 Centralized Error Agent - Logs, analyzes, and suggests fixes")
        print("  🔄 Intelligent Retry Logic - Exponential backoff with strategy selection")
        print("  🧠 Pattern Recognition - Automatic error categorization and clustering")
        print("  💡 Fix Suggestions - AI-powered actionable recommendations")
        print("  📊 Trend Analysis - Long-term error pattern monitoring")
        print("  🛡️  Enterprise Security - Production-ready error management")
        print("  📈 Performance Monitoring - Retry effectiveness and system health")
        print("  🔧 Resolution Tracking - Complete error lifecycle management")
        print()
        print("The enhanced system provides enterprise-grade error handling")
        print("and intelligent recovery capabilities for production environments.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the error handling system capabilities.")
        print("Full functionality requires proper database setup and error simulation.")


if __name__ == "__main__":
    main()