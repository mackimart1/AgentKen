import logging
import time
import json

from core.workflow_monitoring import (
    WorkflowMonitor,
    console_notification_handler,
    log_notification_handler,
)
from core.tool_integration_system import create_tool_system
from core.agent_framework import create_agent_system
from core.adaptive_orchestrator import (
    AdvancedOrchestrator,
    create_sample_plan,
    AgentCapacity,
    ResourceConstraint,
)
from core.legacy_integration import integrate_legacy_components


def main():
    """
    Main function to initialize and run the advanced agent system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- 1. Initialize Workflow Monitoring System ---
    monitor = WorkflowMonitor()
    monitor.alert_manager.add_notification_handler(console_notification_handler)
    monitor.alert_manager.add_notification_handler(log_notification_handler)

    # Add custom alert rules for application-specific metrics
    monitor.alert_manager.add_alert_rule(
        metric_name="app_request_latency",
        threshold=500.0,
        condition="greater_than",
        severity="WARNING",
        description="Application latency is high",
    )
    monitor.alert_manager.add_alert_rule(
        metric_name="app_error_rate",
        threshold=10.0,
        condition="greater_than",
        severity="ERROR",
        description="Application error rate is too high",
    )

    # Start monitoring
    monitor.start()
    logging.info("Workflow monitor started.")

    # --- 2. Initialize Tool Integration System ---
    tool_registry = create_tool_system()
    logging.info(
        "Tool integration system initialized with %d tools.", len(tool_registry.tools)
    )

    # --- 3. Initialize Agent Framework ---
    message_bus, _, agents = create_agent_system()
    logging.info("Agent framework initialized with %d agents.", len(agents))

    # --- 3.5. Integrate Legacy Components ---
    logging.info("Integrating legacy agents and tools...")
    integration_report = integrate_legacy_components(tool_registry, agents, message_bus)
    logging.info("Legacy integration completed:")
    logging.info(
        f"  - Agents: {integration_report['wrapped_agents']}/{integration_report['discovered_agents']}"
    )
    logging.info(
        f"  - Tools: {integration_report['wrapped_tools']}/{integration_report['discovered_tools']}"
    )

    # --- 4. Initialize Advanced Orchestrator ---
    orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

    # Add agent capacities and resource constraints to the orchestrator
    orchestrator.agent_capacities["research_agent"] = AgentCapacity(
        agent_id="research_agent",
        max_concurrent=3,
        capabilities=["web_research", "data_analysis"],
        performance_score=0.9,
    )

    # Add all required resource constraints
    orchestrator.resource_constraints["internet_access"] = ResourceConstraint(
        name="internet_access", capacity=10
    )
    orchestrator.resource_constraints["cpu_intensive"] = ResourceConstraint(
        name="cpu_intensive", capacity=5
    )
    orchestrator.resource_constraints["document_generator"] = ResourceConstraint(
        name="document_generator", capacity=3
    )
    logging.info("Advanced orchestrator initialized.")

    # --- 5. Create and Execute a Plan ---
    plan_data = create_sample_plan()
    plan_id = orchestrator.create_plan(**plan_data)
    logging.info(f"Created execution plan: {plan_id}")

    # Start plan execution
    orchestrator.execute_plan(plan_id)
    logging.info(f"Executing plan: {plan_id}")

    # --- 6. Monitor Execution and Report Status ---
    try:
        # Monitor for a short period
        for i in range(5):
            time.sleep(2)
            plan_status = orchestrator.get_plan_status(plan_id)
            system_metrics = orchestrator.get_system_metrics()

            logging.info(f"Plan Status: {plan_status}")
            logging.info(f"System Metrics: {json.dumps(system_metrics, indent=2)}")

        # Generate and print an optimization report
        optimization_report = monitor.get_optimization_report()
        logging.info("\n=== OPTIMIZATION REPORT ===")
        logging.info(
            f"System Health Score: {optimization_report['system_health_score']:.1f}/100"
        )
        logging.info(f"Active Alerts: {optimization_report['active_alerts_count']}")
        logging.info(
            f"Bottlenecks Detected: {optimization_report['bottlenecks_detected']}"
        )

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        # --- 7. Graceful Shutdown ---
        monitor.stop()
        for agent in agents:
            agent.shutdown()
        logging.info("System shutdown complete.")


if __name__ == "__main__":
    main()
