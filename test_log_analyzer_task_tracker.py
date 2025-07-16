#!/usr/bin/env python3
"""
Comprehensive test suite for LogAnalyzer tool and TaskTracker agent.
Tests functionality, integration, and performance.
"""

import json
import logging
import tempfile
import os
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_log_analyzer():
    """Test LogAnalyzer tool functionality."""
    print("=" * 60)
    print("TESTING LOG ANALYZER TOOL")
    print("=" * 60)
    
    try:
        # Import LogAnalyzer tools
        from tools.log_analyzer import (
            analyze_log_content, 
            analyze_log_file, 
            find_log_files, 
            analyze_error_patterns
        )
        
        # Test 1: Analyze log content
        print("\n1. Testing analyze_log_content...")
        sample_log = """
2025-01-16 10:30:15 INFO main.py Starting application
2025-01-16 10:30:16 DEBUG config.py Loading configuration
2025-01-16 10:30:17 INFO database.py Connected to database
2025-01-16 10:30:18 WARNING auth.py Invalid login attempt from 192.168.1.100
2025-01-16 10:30:19 ERROR payment.py Payment processing failed: Connection timeout
2025-01-16 10:30:20 CRITICAL security.py Security breach detected: Unauthorized access
2025-01-16 10:30:21 ERROR payment.py Payment processing failed: Connection timeout
2025-01-16 10:30:22 INFO main.py Application running normally
2025-01-16 10:30:23 ERROR database.py Database connection lost
2025-01-16 10:30:24 WARNING auth.py Rate limit exceeded for user 12345
        """
        
        result = analyze_log_content(sample_log)
        result_data = json.loads(result)
        
        print(f"Status: {result_data['status']}")
        if result_data['status'] == 'success':
            analysis = result_data['analysis']
            print(f"Total entries: {analysis['total_entries']}")
            print(f"Error count: {analysis['error_count']}")
            print(f"Warning count: {analysis['warning_count']}")
            print(f"Critical count: {analysis['critical_count']}")
            print(f"Unique errors: {len(analysis['unique_errors'])}")
            print(f"Error patterns: {len(analysis['error_patterns'])}")
            print(f"Anomalies detected: {len(analysis['anomalies'])}")
            print(f"Recommendations: {len(analysis['recommendations'])}")
            
            # Print some details
            if analysis['error_patterns']:
                print("\nTop error patterns:")
                for pattern, count in list(analysis['error_patterns'].items())[:3]:
                    print(f"  - {pattern}: {count} occurrences")
            
            if analysis['anomalies']:
                print("\nAnomalies detected:")
                for anomaly in analysis['anomalies'][:2]:
                    print(f"  - {anomaly['type']}: {anomaly['description']}")
        
        # Test 2: Create temporary log file and analyze it
        print("\n2. Testing analyze_log_file...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(sample_log)
            temp_log_path = f.name
        
        try:
            result = analyze_log_file(temp_log_path, max_lines=100)
            result_data = json.loads(result)
            print(f"File analysis status: {result_data['status']}")
            if result_data['status'] == 'success':
                print(f"Lines analyzed: {result_data['lines_analyzed']}")
                print(f"File path: {result_data['file_path']}")
        finally:
            os.unlink(temp_log_path)
        
        # Test 3: Test error pattern analysis
        print("\n3. Testing analyze_error_patterns...")
        result = analyze_error_patterns(sample_log, min_occurrences=2)
        result_data = json.loads(result)
        
        print(f"Pattern analysis status: {result_data['status']}")
        if result_data['status'] == 'success':
            print(f"Total errors: {result_data['total_errors']}")
            print(f"Unique patterns: {result_data['unique_patterns']}")
            print(f"Significant patterns: {result_data['significant_patterns']}")
            
            if result_data['patterns']:
                print("\nSignificant error patterns:")
                for pattern in result_data['patterns'][:2]:
                    print(f"  - Pattern: {pattern['pattern']}")
                    print(f"    Count: {pattern['count']}")
                    print(f"    Sources: {pattern['sources']}")
        
        # Test 4: Test find_log_files
        print("\n4. Testing find_log_files...")
        result = find_log_files(".", "*.log")
        result_data = json.loads(result)
        print(f"Find files status: {result_data['status']}")
        if result_data['status'] == 'success':
            print(f"Files found: {result_data['files_found']}")
        
        print("\n✅ LogAnalyzer tool tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ LogAnalyzer tool test failed: {e}")
        logger.error(f"LogAnalyzer test error: {e}", exc_info=True)
        return False


def test_task_tracker_agent():
    """Test TaskTracker agent functionality."""
    print("\n" + "=" * 60)
    print("TESTING TASK TRACKER AGENT")
    print("=" * 60)
    
    try:
        # Import TaskTracker agent
        from agents.task_tracker_agent import task_tracker_agent, task_tracker
        
        # Test 1: Create tasks
        print("\n1. Testing task creation...")
        
        # Create some test tasks
        task_id1 = task_tracker.create_task(
            "Implement user authentication",
            "Add login/logout functionality with JWT tokens",
            priority="high",
            assigned_to="developer_1",
            tags=["authentication", "security"]
        )
        print(f"Created task: {task_id1}")
        
        task_id2 = task_tracker.create_task(
            "Write unit tests",
            "Create comprehensive unit tests for authentication module",
            priority="normal",
            assigned_to="developer_2",
            tags=["testing", "quality"]
        )
        print(f"Created task: {task_id2}")
        
        task_id3 = task_tracker.create_task(
            "Database optimization",
            "Optimize database queries for better performance",
            priority="low",
            assigned_to="developer_1",
            tags=["performance", "database"]
        )
        print(f"Created task: {task_id3}")
        
        # Test 2: Update task status
        print("\n2. Testing task status updates...")
        success = task_tracker.update_task_status(task_id1, "in_progress")
        print(f"Updated {task_id1} to in_progress: {success}")
        
        success = task_tracker.update_task_status(task_id2, "completed")
        print(f"Updated {task_id2} to completed: {success}")
        
        # Test 3: Get individual task
        print("\n3. Testing task retrieval...")
        task = task_tracker.get_task(task_id1)
        if task:
            print(f"Retrieved task {task_id1}:")
            print(f"  Title: {task['title']}")
            print(f"  Status: {task['status']}")
            print(f"  Priority: {task['priority']}")
            print(f"  Assigned to: {task['assigned_to']}")
            print(f"  Tags: {task['tags']}")
        
        # Test 4: List tasks with filtering
        print("\n4. Testing task listing...")
        all_tasks = task_tracker.list_tasks()
        print(f"Total tasks: {len(all_tasks)}")
        
        pending_tasks = task_tracker.list_tasks(status="pending")
        print(f"Pending tasks: {len(pending_tasks)}")
        
        dev1_tasks = task_tracker.list_tasks(assigned_to="developer_1")
        print(f"Tasks assigned to developer_1: {len(dev1_tasks)}")
        
        # Test 5: Get task summary
        print("\n5. Testing task summary...")
        summary = task_tracker.get_task_summary()
        print(f"Task summary:")
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  By status: {summary['by_status']}")
        print(f"  By priority: {summary['by_priority']}")
        print(f"  Overdue tasks: {summary['overdue_tasks']}")
        
        # Test 6: Test agent interface
        print("\n6. Testing agent interface...")
        
        # Test direct task creation through agent
        result = task_tracker_agent("create task: Fix critical bug in payment system")
        print(f"Agent task creation result: {result}")
        
        # Test task listing through agent
        result = task_tracker_agent("list tasks")
        print(f"Agent task listing result: {result['status']}")
        if result['status'] == 'success':
            print(f"Tasks returned: {len(result['result'])}")
        
        # Test task summary through agent
        result = task_tracker_agent("task summary")
        print(f"Agent task summary result: {result['status']}")
        
        # Test complex request through agent
        result = task_tracker_agent("Show me all high priority tasks that are currently in progress")
        print(f"Complex query result: {result['status']}")
        
        print("\n✅ TaskTracker agent tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ TaskTracker agent test failed: {e}")
        logger.error(f"TaskTracker test error: {e}", exc_info=True)
        return False


def test_integration():
    """Test integration between LogAnalyzer and TaskTracker."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION")
    print("=" * 60)
    
    try:
        from tools.log_analyzer import analyze_log_content
        from agents.task_tracker_agent import task_tracker_agent, task_tracker
        
        # Test scenario: Analyze logs and create tasks based on findings
        print("\n1. Testing log analysis to task creation workflow...")
        
        # Sample log with errors that should trigger task creation
        error_log = """
2025-01-16 10:30:15 ERROR payment.py Payment gateway timeout - multiple failures
2025-01-16 10:30:16 CRITICAL security.py SQL injection attempt detected
2025-01-16 10:30:17 ERROR database.py Connection pool exhausted
2025-01-16 10:30:18 WARNING performance.py High memory usage detected: 95%
2025-01-16 10:30:19 ERROR payment.py Payment gateway timeout - multiple failures
2025-01-16 10:30:20 ERROR api.py Rate limit exceeded for endpoint /api/users
        """
        
        # Analyze the logs
        result = analyze_log_content(error_log)
        analysis_data = json.loads(result)
        
        if analysis_data['status'] == 'success':
            analysis = analysis_data['analysis']
            print(f"Log analysis completed:")
            print(f"  Errors found: {analysis['error_count']}")
            print(f"  Critical issues: {analysis['critical_count']}")
            print(f"  Anomalies: {len(analysis['anomalies'])}")
            
            # Create tasks based on critical findings
            tasks_created = []
            
            # Create task for critical security issue
            if analysis['critical_count'] > 0:
                task_id = task_tracker.create_task(
                    "URGENT: Security breach investigation",
                    "Investigate and resolve critical security issues found in logs",
                    priority="critical",
                    assigned_to="security_team",
                    tags=["security", "critical", "urgent"]
                )
                tasks_created.append(task_id)
                print(f"  Created critical security task: {task_id}")
            
            # Create task for payment issues
            if any("payment" in pattern.lower() for pattern in analysis['error_patterns']):
                task_id = task_tracker.create_task(
                    "Fix payment gateway issues",
                    "Resolve payment gateway timeout issues affecting transactions",
                    priority="high",
                    assigned_to="payment_team",
                    tags=["payment", "gateway", "timeout"]
                )
                tasks_created.append(task_id)
                print(f"  Created payment issue task: {task_id}")
            
            # Create task for database issues
            if any("database" in pattern.lower() for pattern in analysis['error_patterns']):
                task_id = task_tracker.create_task(
                    "Database connection optimization",
                    "Optimize database connection pool to prevent exhaustion",
                    priority="high",
                    assigned_to="database_team",
                    tags=["database", "performance", "connections"]
                )
                tasks_created.append(task_id)
                print(f"  Created database issue task: {task_id}")
            
            print(f"\nTotal tasks created from log analysis: {len(tasks_created)}")
            
            # Verify tasks were created
            summary = task_tracker.get_task_summary()
            print(f"Updated task summary: {summary['total_tasks']} total tasks")
        
        # Test 2: Use TaskTracker agent to manage log-derived tasks
        print("\n2. Testing task management for log-derived issues...")
        
        result = task_tracker_agent("Show me all critical and high priority tasks")
        print(f"High priority tasks query: {result['status']}")
        
        result = task_tracker_agent("List all tasks tagged with security")
        print(f"Security tasks query: {result['status']}")
        
        print("\n✅ Integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.error(f"Integration test error: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("Starting comprehensive LogAnalyzer and TaskTracker tests...")
    print(f"Test started at: {datetime.now()}")
    
    results = {
        'log_analyzer': False,
        'task_tracker': False,
        'integration': False
    }
    
    # Run individual component tests
    results['log_analyzer'] = test_log_analyzer()
    results['task_tracker'] = test_task_tracker_agent()
    
    # Run integration tests
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! LogAnalyzer and TaskTracker are ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    print(f"Test completed at: {datetime.now()}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)