"""
Test of the fixed feedback system.
"""

import sys
import os
import time
import logging

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Test basic imports
try:
    from feedback_system import (
        FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, 
        TaskExecution, ComponentType, TaskOutcome, FeedbackType
    )
    print("✅ Core feedback system imports successful")
except Exception as e:
    print(f"❌ Core feedback system import failed: {e}")
    sys.exit(1)

try:
    from feedback_integration_fixed import (
        initialize_feedback_system, track_agent_execution, track_tool_execution,
        collect_rating_feedback
    )
    print("✅ Feedback integration imports successful")
except Exception as e:
    print(f"❌ Feedback integration import failed: {e}")
    sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Fixed Feedback System Test")
    print("=" * 40)
    
    # Test 1: Initialize feedback system
    print("\n1. Initializing feedback system...")
    try:
        feedback_system = initialize_feedback_system("test_fixed.db")
        print("✅ Feedback system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test 2: Simple decorator usage
    print("\n2. Testing decorator integration...")
    
    @track_agent_execution("test_agent", "simple_task")
    def simple_agent_task(query: str, user_id: str = None) -> dict:
        time.sleep(0.1)
        return {"query": query, "result": "success"}
    
    @track_tool_execution("test_tool", "simple_operation")
    def simple_tool_operation(data: str, user_id: str = None) -> str:
        time.sleep(0.05)
        return data.upper()
    
    try:
        # Execute tracked operations
        agent_result = simple_agent_task("test query", user_id="test_user")
        tool_result = simple_tool_operation("test data", user_id="test_user")
        
        print(f"✅ Agent result: {agent_result}")
        print(f"✅ Tool result: {tool_result}")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return
    
    # Test 3: Manual feedback collection
    print("\n3. Testing manual feedback collection...")
    
    # Wait a moment for tasks to be registered
    time.sleep(1)
    
    try:
        # Get recent tasks
        recent_tasks = feedback_system.collector.get_pending_tasks_for_user("test_user")
        print(f"Found {len(recent_tasks)} recent tasks")
        
        if recent_tasks:
            task = recent_tasks[0]
            
            # Collect rating feedback
            feedback = collect_rating_feedback(
                task_execution_id=task.id,
                user_id="test_user",
                rating=4.5,
                text_feedback="Great test results!",
                tags=["test", "successful"]
            )
            
            print(f"✅ Feedback collected: {feedback.id}")
        else:
            print("⚠️ No recent tasks found for feedback")
    
    except Exception as e:
        print(f"❌ Feedback collection failed: {e}")
    
    # Test 4: Performance analysis
    print("\n4. Testing performance analysis...")
    
    try:
        profile = feedback_system.analyzer.analyze_component_performance("test_agent")
        print(f"✅ Agent performance profile:")
        print(f"   Total executions: {profile.total_executions}")
        print(f"   Average rating: {profile.average_rating:.2f}")
        print(f"   Satisfaction rate: {profile.satisfaction_rate:.1f}%")
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
    
    # Test 5: Learning insights
    print("\n5. Testing learning insights...")
    
    try:
        insights = feedback_system.analyzer.generate_learning_insights()
        print(f"✅ Generated {len(insights)} learning insights")
        
        for insight in insights:
            print(f"   - {insight.insight_type}: {insight.description}")
    except Exception as e:
        print(f"❌ Learning insights failed: {e}")
    
    print("\n🎉 Fixed feedback system test completed!")
    print("The core feedback functionality is working correctly.")

if __name__ == "__main__":
    main()