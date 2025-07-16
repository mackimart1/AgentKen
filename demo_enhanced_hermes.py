#!/usr/bin/env python3
"""
Demonstration script for Enhanced Hermes capabilities:
1. Context Awareness
2. Dynamic Plan Adaptation  
3. Multi-Tasking

This script shows how the enhanced features work without requiring full LLM integration.
"""

import json
import datetime
from agents.hermes_enhanced import (
    TaskScheduler, PlanAdapter, ContextManager, 
    Task, TaskPriority, TaskStatus, ExecutionPlan, PlanStep
)
from tools.task_manager import create_task, update_task, list_tasks, get_next_task
from tools.context_manager_tool import save_context, load_context, update_user_preference
from tools.plan_adapter_tool import create_adaptive_plan, adapt_plan, analyze_plan_execution


def demo_context_awareness():
    """Demonstrate context awareness capabilities."""
    print("=" * 80)
    print("CONTEXT AWARENESS DEMONSTRATION")
    print("=" * 80)
    
    session_id = "demo_session_001"
    
    # Save some context
    print("1. Saving user preferences and patterns...")
    
    # Save user preference
    pref_result = update_user_preference.invoke({
        "preference_key": "preferred_agent",
        "preference_value": "software_engineer",
        "session_id": session_id
    })
    print(f"   Preference saved: {json.loads(pref_result)['message']}")
    
    # Save successful pattern
    pattern_result = save_context.invoke({
        "session_id": session_id,
        "context_type": "pattern",
        "data": {
            "pattern": "code_development_workflow",
            "success": True,
            "description": "User prefers step-by-step code development with testing",
            "agents_used": ["software_engineer", "code_executor"],
            "duration": 45
        },
        "importance": 8
    })
    print(f"   Pattern saved: {json.loads(pattern_result)['message']}")
    
    # Save goal context
    goal_result = save_context.invoke({
        "session_id": session_id,
        "context_type": "goal",
        "data": {
            "goal": "Create a web application",
            "completed": True,
            "satisfaction": 9,
            "feedback": "Great job, very thorough approach"
        },
        "importance": 7
    })
    print(f"   Goal saved: {json.loads(goal_result)['message']}")
    
    print("\n2. Loading context for new session...")
    
    # Load context
    context_result = load_context.invoke({
        "session_id": session_id,
        "limit": 5
    })
    context_data = json.loads(context_result)
    
    print(f"   Loaded {context_data['count']} context items:")
    for i, context in enumerate(context_data['contexts'][:3], 1):
        print(f"     {i}. {context['context_type']}: {context['data']}")
    
    print("\n3. Context-aware recommendations...")
    context_manager = ContextManager()
    context_manager.current_context = {
        "agent_performance": {
            "software_engineer_code_development": {
                "successes": 8,
                "failures": 1,
                "avg_duration": 30
            },
            "web_researcher_research": {
                "successes": 5,
                "failures": 2,
                "avg_duration": 20
            }
        }
    }
    
    recommended_agent = context_manager.get_agent_recommendation("code_development")
    print(f"   Recommended agent for code development: {recommended_agent}")
    
    return context_data


def demo_multi_tasking():
    """Demonstrate multi-tasking capabilities."""
    print("\n" + "=" * 80)
    print("MULTI-TASKING DEMONSTRATION")
    print("=" * 80)
    
    # Create multiple tasks with different priorities
    print("1. Creating multiple tasks with different priorities...")
    
    tasks = [
        {
            "description": "Fix critical security vulnerability",
            "priority": "CRITICAL",
            "estimated_duration": 60
        },
        {
            "description": "Implement new user authentication",
            "priority": "HIGH", 
            "estimated_duration": 120
        },
        {
            "description": "Update documentation",
            "priority": "LOW",
            "estimated_duration": 30
        },
        {
            "description": "Optimize database queries",
            "priority": "NORMAL",
            "estimated_duration": 90
        },
        {
            "description": "Write unit tests",
            "priority": "HIGH",
            "dependencies": [],  # Will be updated after creating auth task
            "estimated_duration": 45
        }
    ]
    
    created_task_ids = []
    for task_data in tasks:
        result = create_task.invoke(task_data)
        result_data = json.loads(result)
        if result_data["status"] == "success":
            task_id = result_data["task_id"]
            created_task_ids.append(task_id)
            print(f"   Created: {task_data['description']} (ID: {task_id}, Priority: {task_data['priority']})")
    
    # Update dependencies
    if len(created_task_ids) >= 5:
        update_task.invoke({
            "task_id": created_task_ids[4],  # Unit tests task
            "status": "pending"
        })
        print(f"   Updated dependencies for unit tests task")
    
    print("\n2. Listing tasks by priority...")
    
    # List all tasks
    all_tasks_result = list_tasks.invoke({"include_completed": False})
    all_tasks_data = json.loads(all_tasks_result)
    
    print(f"   Total pending tasks: {all_tasks_data['count']}")
    for task in all_tasks_data['tasks'][:5]:
        print(f"     - {task['description'][:40]}... (Priority: {task['priority']}, Duration: {task.get('estimated_duration', 'N/A')}min)")
    
    print("\n3. Getting next task to execute...")
    
    # Get next task based on priority
    next_task_result = get_next_task.invoke({})
    next_task_data = json.loads(next_task_result)
    
    if next_task_data["task"]:
        task = next_task_data["task"]
        print(f"   Next task: {task['description']}")
        print(f"   Priority: {task['priority']}")
        print(f"   Estimated duration: {task.get('estimated_duration', 'N/A')} minutes")
        
        # Simulate task execution
        print("\n4. Simulating task execution...")
        update_result = update_task.invoke({
            "task_id": task["id"],
            "status": "completed",
            "result": "Task completed successfully"
        })
        print(f"   {json.loads(update_result)['message']}")
    
    return created_task_ids


def demo_plan_adaptation():
    """Demonstrate dynamic plan adaptation."""
    print("\n" + "=" * 80)
    print("DYNAMIC PLAN ADAPTATION DEMONSTRATION")
    print("=" * 80)
    
    # Create an adaptive plan
    print("1. Creating adaptive plan...")
    
    plan_result = create_adaptive_plan.invoke({
        "goal": "Build a complete web application with user authentication",
        "constraints": ["Must be completed in 2 weeks", "Use Python/Flask"],
        "available_agents": ["software_engineer", "web_researcher", "security_agent"],
        "priority": "HIGH"
    })
    
    plan_data = json.loads(plan_result)
    if plan_data["status"] == "success":
        plan = plan_data["plan"]
        print(f"   Plan created: {plan['id']}")
        print(f"   Goal: {plan['goal']}")
        print(f"   Steps: {len(plan['steps'])}")
        print(f"   Estimated duration: {plan['estimated_duration']} minutes")
        
        for i, step in enumerate(plan['steps'], 1):
            print(f"     {i}. {step['description']} (Agent: {step.get('agent', 'TBD')}, Duration: {step.get('estimated_duration', 'N/A')}min)")
    
    print("\n2. Simulating execution feedback...")
    
    # Simulate execution feedback with some failures
    execution_feedback = [
        {
            "step_index": 0,
            "status": "success",
            "duration": 18,
            "step_description": "Analyze requirements and gather resources"
        },
        {
            "step_index": 1,
            "status": "failure",
            "duration": 45,
            "step_description": "Design and plan implementation",
            "message": "Agent software_engineer not available - timeout error"
        },
        {
            "step_index": 2,
            "status": "pending",
            "step_description": "Implement and test solution"
        }
    ]
    
    print("   Execution feedback:")
    for feedback in execution_feedback:
        status_icon = "✅" if feedback["status"] == "success" else "❌" if feedback["status"] == "failure" else "⏳"
        print(f"     {status_icon} Step {feedback['step_index']}: {feedback['status']} ({feedback.get('duration', 'N/A')}min)")
        if feedback.get("message"):
            print(f"        Error: {feedback['message']}")
    
    print("\n3. Analyzing plan execution...")
    
    # Analyze execution
    analysis_result = analyze_plan_execution.invoke({
        "plan_text": "Original plan with 3 steps for web application development",
        "execution_feedback": execution_feedback,
        "goal": "Build a complete web application with user authentication"
    })
    
    analysis_data = json.loads(analysis_result)
    if analysis_data["status"] == "success":
        analysis = analysis_data["analysis"]
        print(f"   Success rate: {analysis['success_rate']:.1%}")
        print(f"   Risk assessment: {analysis['risk_assessment']}")
        print(f"   Failure points: {len(analysis['failure_points'])}")
        
        if analysis["adaptation_recommendations"]:
            print("   Adaptation recommendations:")
            for rec in analysis["adaptation_recommendations"]:
                print(f"     - {rec['type']}: {rec['description']} (Priority: {rec['priority']})")
    
    print("\n4. Adapting plan based on failures...")
    
    # Adapt the plan
    adaptation_result = adapt_plan.invoke({
        "original_plan": "Step 1: Analyze requirements\nStep 2: Design and plan implementation\nStep 3: Implement and test solution",
        "failure_reason": "Agent software_engineer not available - timeout error",
        "failed_step": "Step 2: Design and plan implementation",
        "adaptation_strategy": "agent_substitution"
    })
    
    adaptation_data = json.loads(adaptation_result)
    if adaptation_data["status"] == "success":
        print(f"   Adaptation applied: {adaptation_data['strategy_applied']}")
        print(f"   Adaptation ID: {adaptation_data['adaptation_id']}")
        print("   Adapted plan:")
        adapted_lines = adaptation_data['adapted_plan'].split('\n')
        for line in adapted_lines:
            if line.strip():
                print(f"     {line}")
    
    return plan_data, analysis_data, adaptation_data


def demo_integration():
    """Demonstrate how all three capabilities work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED CAPABILITIES DEMONSTRATION")
    print("=" * 80)
    
    print("1. Enhanced Hermes workflow simulation...")
    
    # Simulate a complete enhanced workflow
    session_id = "integrated_demo_session"
    
    # Context loading
    print("   📚 Loading context from previous sessions...")
    context_result = load_context.invoke({
        "session_id": session_id,
        "context_type": "preference",
        "limit": 3
    })
    context_data = json.loads(context_result)
    print(f"      Loaded {context_data['count']} context items")
    
    # Multi-task creation based on context
    print("   📋 Creating prioritized tasks based on user preferences...")
    high_priority_task = create_task.invoke({
        "description": "Implement user-requested feature based on previous feedback",
        "priority": "HIGH",
        "assigned_agent": "software_engineer",  # Based on context preference
        "estimated_duration": 60,
        "context": {"user_preference": "step_by_step_development"}
    })
    task_data = json.loads(high_priority_task)
    print(f"      Created high-priority task: {task_data.get('task_id', 'N/A')}")
    
    # Plan creation with adaptation capability
    print("   🎯 Creating adaptive plan with context awareness...")
    adaptive_plan = create_adaptive_plan.invoke({
        "goal": "Complete user-requested feature with high quality",
        "available_agents": ["software_engineer", "security_agent"],
        "constraints": ["Follow user's preferred step-by-step approach"],
        "priority": "HIGH"
    })
    plan_data = json.loads(adaptive_plan)
    print(f"      Created adaptive plan: {plan_data.get('plan_id', 'N/A')}")
    
    # Simulate real-time adaptation
    print("   🔄 Simulating real-time plan adaptation...")
    if plan_data["status"] == "success":
        # Simulate a failure and adaptation
        adaptation = adapt_plan.invoke({
            "original_plan": "Step 1: Analyze\nStep 2: Implement\nStep 3: Test",
            "failure_reason": "User changed requirements mid-execution",
            "failed_step": "Step 2: Implement",
            "adaptation_strategy": "step_modification"
        })
        adaptation_data = json.loads(adaptation)
        print(f"      Plan adapted: {adaptation_data.get('strategy_applied', 'N/A')}")
    
    # Context saving for future sessions
    print("   💾 Saving insights for future sessions...")
    insight_result = save_context.invoke({
        "session_id": session_id,
        "context_type": "insight",
        "data": {
            "insight": "User prefers adaptive plans that can handle requirement changes",
            "confidence": 0.9,
            "applicable_scenarios": ["feature_development", "requirement_changes"]
        },
        "importance": 8
    })
    insight_data = json.loads(insight_result)
    print(f"      Insight saved: {insight_data.get('context_id', 'N/A')}")
    
    print("\n2. Enhanced capabilities summary:")
    print("   ✅ Context Awareness: Loaded preferences and patterns from previous sessions")
    print("   ✅ Multi-Tasking: Created and prioritized tasks based on context")
    print("   ✅ Plan Adaptation: Created adaptive plan and demonstrated real-time modification")
    print("   ✅ Learning: Saved insights for future session improvement")
    
    return {
        "context_loaded": context_data['count'],
        "tasks_created": 1,
        "plans_created": 1,
        "adaptations_made": 1,
        "insights_saved": 1
    }


def main():
    """Run all demonstrations."""
    print("ENHANCED HERMES CAPABILITIES DEMONSTRATION")
    print("This demo shows the three key improvements:")
    print("1. Context Awareness - Cross-session memory and learning")
    print("2. Dynamic Plan Adaptation - Real-time plan modification")
    print("3. Multi-Tasking - Priority-based task scheduling")
    print()
    
    try:
        # Run individual demonstrations
        context_demo = demo_context_awareness()
        task_demo = demo_multi_tasking()
        plan_demo = demo_plan_adaptation()
        integration_demo = demo_integration()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print(f"✅ Context items managed: {context_demo['count']}")
        print(f"✅ Tasks created and managed: {len(task_demo)}")
        print(f"✅ Plans created and adapted: 1")
        print(f"✅ Integration scenarios: 1")
        print()
        print("Enhanced Hermes is ready with:")
        print("  🧠 Context Awareness - Remembers and learns from past interactions")
        print("  🔄 Dynamic Adaptation - Modifies plans in real-time based on feedback")
        print("  📋 Multi-Tasking - Manages multiple priorities simultaneously")
        print()
        print("The enhanced system provides a more intelligent, adaptive, and")
        print("context-aware orchestration experience for users.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the enhanced capabilities structure.")
        print("Full integration requires the complete enhanced Hermes system.")


if __name__ == "__main__":
    main()