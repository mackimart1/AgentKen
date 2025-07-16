#!/usr/bin/env python3
"""
Simple verification script for LogAnalyzer and TaskTracker implementations.
"""

import json
import sys

def test_log_analyzer():
    """Test LogAnalyzer basic functionality."""
    print("Testing LogAnalyzer...")
    
    try:
        from tools.log_analyzer import analyze_log_content
        
        # Simple test
        test_log = "2025-01-16 10:30:15 ERROR test.py Test error message"
        result = analyze_log_content(test_log)
        
        # Parse result
        data = json.loads(result)
        
        if data['status'] == 'success':
            print("✅ LogAnalyzer: Basic functionality working")
            analysis = data.get('analysis', {})
            print(f"   - Total entries: {analysis.get('total_entries', 0)}")
            print(f"   - Error count: {analysis.get('error_count', 0)}")
            return True
        else:
            print(f"❌ LogAnalyzer: Failed - {data.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ LogAnalyzer: Exception - {e}")
        return False


def test_task_tracker():
    """Test TaskTracker basic functionality."""
    print("\nTesting TaskTracker...")
    
    try:
        from agents.task_tracker_agent import task_tracker
        
        # Create a test task
        task_id = task_tracker.create_task(
            "Test task",
            "This is a test task",
            priority="normal"
        )
        
        if task_id:
            print("✅ TaskTracker: Task creation working")
            print(f"   - Created task: {task_id}")
            
            # Get task
            task = task_tracker.get_task(task_id)
            if task:
                print(f"   - Task retrieval working: {task['title']}")
                
                # Get summary
                summary = task_tracker.get_task_summary()
                print(f"   - Summary working: {summary['total_tasks']} total tasks")
                
                return True
            else:
                print("❌ TaskTracker: Task retrieval failed")
                return False
        else:
            print("❌ TaskTracker: Task creation failed")
            return False
            
    except Exception as e:
        print(f"❌ TaskTracker: Exception - {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 50)
    print("VERIFYING LOGANALYZER AND TASKTRACKER")
    print("=" * 50)
    
    log_analyzer_ok = test_log_analyzer()
    task_tracker_ok = test_task_tracker()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"LogAnalyzer: {'✅ WORKING' if log_analyzer_ok else '❌ FAILED'}")
    print(f"TaskTracker: {'✅ WORKING' if task_tracker_ok else '❌ FAILED'}")
    
    if log_analyzer_ok and task_tracker_ok:
        print("\n🎉 Both implementations are working correctly!")
        return True
    else:
        print("\n⚠️ Some implementations have issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)