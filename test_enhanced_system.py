#!/usr/bin/env python3
"""
Test script for the Enhanced Inferra V Agent System
Demonstrates key features and capabilities of the improved system.
"""

import logging
import time
import json
from core.workflow_monitoring import WorkflowMonitor, AlertSeverity
from core.tool_integration_system import create_tool_system
from core.agent_framework import create_agent_system
from core.adaptive_orchestrator import (
    AdvancedOrchestrator,
    TaskPriority,
    AgentCapacity,
    ResourceConstraint,
)


def test_basic_functionality():
    """Test basic system functionality"""
    print("🚀 Testing Enhanced Inferra V System")
    print("=" * 50)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize systems
    print("📊 Initializing monitoring system...")
    monitor = WorkflowMonitor()
    monitor.start()

    print("🛠️ Initializing tool system...")
    tool_registry = create_tool_system()

    print("🤖 Initializing agent framework...")
    message_bus, _, agents = create_agent_system()

    print("🎯 Initializing orchestrator...")
    orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

    # Configure system
    orchestrator.agent_capacities["research_agent"] = AgentCapacity(
        agent_id="research_agent",
        max_concurrent=3,
        capabilities=["web_search", "data_processing"],
        performance_score=0.9,
    )

    orchestrator.resource_constraints.update(
        {
            "network_access": ResourceConstraint("network_access", 10),
            "cpu_intensive": ResourceConstraint("cpu_intensive", 5),
            "storage": ResourceConstraint("storage", 100),
        }
    )

    return monitor, orchestrator, agents


def test_plan_execution(orchestrator):
    """Test plan creation and execution"""
    print("\n📋 Testing Plan Execution")
    print("-" * 30)

    # Create a test plan
    tasks = [
        {
            "id": "search_task",
            "name": "Web Search",
            "agent_type": "research_agent",
            "capability": "web_search",
            "parameters": {"query": "AI trends 2024", "max_results": 5},
            "priority": TaskPriority.HIGH.value,
            "estimated_duration": 3.0,
            "dependencies": [],
            "required_resources": ["network_access"],
        },
        {
            "id": "process_task",
            "name": "Data Processing",
            "agent_type": "research_agent",
            "capability": "data_processing",
            "parameters": {"data": {"test": "data"}, "operation": "analyze"},
            "priority": TaskPriority.NORMAL.value,
            "estimated_duration": 2.0,
            "dependencies": ["search_task"],
            "required_resources": ["cpu_intensive"],
        },
    ]

    plan_id = orchestrator.create_plan(
        name="Test Analysis Plan",
        description="Test plan for system validation",
        tasks=tasks,
        deadline=time.time() + 300,  # 5 minutes
    )

    print(f"✅ Created plan: {plan_id}")

    # Execute plan
    success = orchestrator.execute_plan(plan_id)
    print(f"✅ Plan execution started: {success}")

    # Monitor execution
    for i in range(5):
        time.sleep(2)
        status = orchestrator.get_plan_status(plan_id)
        if status:
            print(
                f"📊 Progress: {status['progress']:.1%} ({status['completed_tasks']}/{status['total_tasks']} tasks)"
            )
            if status["status"] in ["completed", "failed"]:
                break

    return plan_id


def test_monitoring_features(monitor):
    """Test monitoring and alerting features"""
    print("\n📈 Testing Monitoring Features")
    print("-" * 30)

    # Add custom alert
    monitor.alert_manager.add_alert_rule(
        metric_name="test_metric",
        threshold=50.0,
        condition="greater_than",
        severity=AlertSeverity.INFO,
        description="Test metric for demonstration",
    )

    # Simulate some metrics
    from core.workflow_monitoring import Metric, MetricType

    test_metric = Metric(
        name="test_metric",
        type=MetricType.THROUGHPUT,
        value=75.0,  # Above threshold
        timestamp=time.time(),
        unit="requests_per_second",
    )

    monitor.metrics_collector.record_metric(test_metric)

    # Wait for alert processing
    time.sleep(3)

    # Check alerts
    active_alerts = monitor.alert_manager.get_active_alerts()
    print(f"🚨 Active alerts: {len(active_alerts)}")

    # Get system health
    health_score = monitor.get_system_health_score()
    print(f"💚 System health score: {health_score:.1f}/100")


def test_optimization_features(monitor):
    """Test optimization and recommendation features"""
    print("\n🔧 Testing Optimization Features")
    print("-" * 30)

    # Generate optimization report
    report = monitor.get_optimization_report()

    print(f"📊 System Health: {report['system_health_score']:.1f}/100")
    print(f"🚨 Active Alerts: {report['active_alerts_count']}")
    print(f"🔍 Bottlenecks: {report['bottlenecks_detected']}")
    print(f"💡 Recommendations: {len(report['optimization_recommendations'])}")

    # Show top recommendations
    if report["optimization_recommendations"]:
        print("\n🎯 Top Recommendations:")
        for i, rec in enumerate(report["optimization_recommendations"][:3], 1):
            print(f"  {i}. {rec.title}")
            print(f"     Expected improvement: {rec.expected_improvement:.1f}%")
            print(f"     Priority: {rec.priority_score:.1f}")


def test_tool_integration(tool_registry):
    """Test tool integration features"""
    print("\n🛠️ Testing Tool Integration")
    print("-" * 30)

    # Test tool execution
    try:
        result = tool_registry.execute_tool(
            "web_search", query="test query", max_results=3
        )
        print("✅ Web search tool executed successfully")
        print(f"📄 Result: {result['results'][:2]}...")  # Show first 2 results
    except Exception as e:
        print(f"❌ Tool execution failed: {e}")

    # Get system status
    status = tool_registry.get_system_status()
    print(f"🔧 Available tools: {status['available_tools']}")
    print(f"📊 Total tools: {status['total_tools']}")


def main():
    """Main test function"""
    try:
        # Initialize systems
        monitor, orchestrator, agents = test_basic_functionality()

        # Test core features
        plan_id = test_plan_execution(orchestrator)
        test_monitoring_features(monitor)
        test_optimization_features(monitor)
        test_tool_integration(orchestrator.tool_registry)

        # Final system metrics
        print("\n📊 Final System Metrics")
        print("-" * 30)
        metrics = orchestrator.get_system_metrics()
        print(json.dumps(metrics, indent=2))

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            monitor.stop()
            for agent in agents:
                agent.shutdown()
            print("✅ Cleanup completed")
        except:
            pass


if __name__ == "__main__":
    main()
