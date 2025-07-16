"""
Setup script for integrating performance monitoring into AgentKen system.
This script configures and initializes the performance monitoring layer.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from performance_monitor import PerformanceMonitor, AlertLevel, MetricType
from performance_decorators import initialize_performance_monitoring, integrate_with_existing_agent, integrate_with_existing_tool
from performance_dashboard import PerformanceDashboardServer, create_dashboard_template


class PerformanceMonitoringSetup:
    """Setup and configuration for performance monitoring"""
    
    def __init__(self, config_path: str = "performance_config.json"):
        self.config_path = config_path
        self.config = self._load_or_create_config()
        self.monitor = None
        self.dashboard_server = None
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load configuration or create default"""
        default_config = {
            "database": {
                "path": "performance_metrics.db",
                "retention_hours": 168  # 7 days
            },
            "alerts": {
                "enabled": True,
                "rules": [
                    {
                        "component_id": "*",
                        "metric_type": "latency",
                        "threshold": 5000.0,
                        "condition": "greater_than",
                        "level": "warning",
                        "description": "High latency detected"
                    },
                    {
                        "component_id": "*",
                        "metric_type": "success_rate",
                        "threshold": 90.0,
                        "condition": "less_than",
                        "level": "error",
                        "description": "Low success rate detected"
                    },
                    {
                        "component_id": "*",
                        "metric_type": "failure_rate",
                        "threshold": 10.0,
                        "condition": "greater_than",
                        "level": "warning",
                        "description": "High failure rate detected"
                    }
                ]
            },
            "dashboard": {
                "enabled": True,
                "host": "localhost",
                "port": 5001,
                "auto_refresh_seconds": 30
            },
            "monitoring": {
                "enabled": True,
                "auto_discover_components": True,
                "track_system_metrics": True,
                "track_agent_metrics": True,
                "track_tool_metrics": True
            },
            "notifications": {
                "console": True,
                "log": True,
                "email": False,
                "webhook": False
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**default_config, **config}
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
                logging.info("Using default configuration")
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def initialize_monitoring(self) -> PerformanceMonitor:
        """Initialize the performance monitoring system"""
        if not self.config["monitoring"]["enabled"]:
            logging.info("Performance monitoring is disabled in configuration")
            return None
        
        db_path = self.config["database"]["path"]
        self.monitor = initialize_performance_monitoring(db_path)
        
        # Setup alert rules
        if self.config["alerts"]["enabled"]:
            self._setup_alert_rules()
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        logging.info("Performance monitoring initialized successfully")
        return self.monitor
    
    def _setup_alert_rules(self):
        """Setup alert rules from configuration"""
        for rule in self.config["alerts"]["rules"]:
            try:
                metric_type = MetricType(rule["metric_type"])
                level = AlertLevel(rule["level"])
                
                self.monitor.alert_manager.add_alert_rule(
                    component_id=rule["component_id"],
                    metric_type=metric_type,
                    threshold=rule["threshold"],
                    condition=rule["condition"],
                    level=level,
                    description=rule["description"]
                )
                
                logging.info(f"Added alert rule: {rule['component_id']} - {rule['metric_type']}")
            except Exception as e:
                logging.error(f"Failed to add alert rule: {e}")
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        from performance_monitor import console_notification_handler, log_notification_handler
        
        if self.config["notifications"]["console"]:
            self.monitor.alert_manager.add_notification_handler(console_notification_handler)
        
        if self.config["notifications"]["log"]:
            self.monitor.alert_manager.add_notification_handler(log_notification_handler)
        
        # Add custom notification handlers here if needed
        if self.config["notifications"]["email"]:
            # self.monitor.alert_manager.add_notification_handler(email_notification_handler)
            logging.warning("Email notifications not implemented yet")
        
        if self.config["notifications"]["webhook"]:
            # self.monitor.alert_manager.add_notification_handler(webhook_notification_handler)
            logging.warning("Webhook notifications not implemented yet")
    
    def setup_dashboard(self) -> PerformanceDashboardServer:
        """Setup the performance dashboard"""
        if not self.config["dashboard"]["enabled"] or not self.monitor:
            logging.info("Dashboard is disabled or monitoring not initialized")
            return None
        
        # Create templates directory and file
        self._create_dashboard_template()
        
        # Create dashboard server
        self.dashboard_server = PerformanceDashboardServer(
            self.monitor,
            host=self.config["dashboard"]["host"],
            port=self.config["dashboard"]["port"]
        )
        
        logging.info(f"Dashboard setup complete. Access at http://{self.config['dashboard']['host']}:{self.config['dashboard']['port']}")
        return self.dashboard_server
    
    def _create_dashboard_template(self):
        """Create the dashboard HTML template"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        template_path = os.path.join(templates_dir, 'performance_dashboard.html')
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write(create_dashboard_template())
            logging.info(f"Dashboard template created at {template_path}")
    
    def integrate_with_agents(self, agents_manifest_path: str = "agents_manifest.json"):
        """Integrate monitoring with existing agents"""
        if not self.config["monitoring"]["track_agent_metrics"]:
            return
        
        try:
            if os.path.exists(agents_manifest_path):
                with open(agents_manifest_path, 'r') as f:
                    agents_manifest = json.load(f)
                
                for agent_id, agent_info in agents_manifest.items():
                    logging.info(f"Integrating monitoring with agent: {agent_id}")
                    # Here you would integrate with actual agent classes
                    # This is a placeholder for the integration logic
        except Exception as e:
            logging.error(f"Failed to integrate with agents: {e}")
    
    def integrate_with_tools(self, tools_manifest_path: str = "tools_manifest.json"):
        """Integrate monitoring with existing tools"""
        if not self.config["monitoring"]["track_tool_metrics"]:
            return
        
        try:
            if os.path.exists(tools_manifest_path):
                with open(tools_manifest_path, 'r') as f:
                    tools_manifest = json.load(f)
                
                for tool_id, tool_info in tools_manifest.items():
                    logging.info(f"Integrating monitoring with tool: {tool_id}")
                    # Here you would integrate with actual tool classes
                    # This is a placeholder for the integration logic
        except Exception as e:
            logging.error(f"Failed to integrate with tools: {e}")
    
    def run_dashboard(self, debug: bool = False):
        """Run the dashboard server"""
        if self.dashboard_server:
            self.dashboard_server.run(debug=debug)
        else:
            logging.error("Dashboard server not initialized")
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate a setup report"""
        return {
            "monitoring_enabled": self.monitor is not None,
            "dashboard_enabled": self.dashboard_server is not None,
            "config_path": self.config_path,
            "database_path": self.config["database"]["path"],
            "alert_rules_count": len(self.config["alerts"]["rules"]),
            "dashboard_url": f"http://{self.config['dashboard']['host']}:{self.config['dashboard']['port']}" if self.dashboard_server else None,
            "configuration": self.config
        }


def create_example_integration():
    """Create example integration with AgentKen components"""
    
    # Example agent integration
    example_agent_code = '''
from core.performance_decorators import monitor_agent_execution, MonitoredAgent
import time

class ExampleAgent(MonitoredAgent):
    """Example agent with performance monitoring"""
    
    def __init__(self):
        super().__init__("example_agent")
    
    def research_topic(self, topic: str) -> dict:
        """Research a topic with monitoring"""
        return self.execute_with_monitoring("research_topic", self._do_research, topic)
    
    def _do_research(self, topic: str) -> dict:
        # Simulate research work
        time.sleep(0.2)
        return {"topic": topic, "results": ["result1", "result2"]}

# Alternative using decorators
@monitor_agent_execution("research_agent", "web_search")
def search_web(query: str) -> dict:
    time.sleep(0.1)
    return {"query": query, "results": ["web result 1", "web result 2"]}
'''
    
    # Example tool integration
    example_tool_code = '''
from core.performance_decorators import monitor_tool_execution, MonitoredTool
import time

class ExampleTool(MonitoredTool):
    """Example tool with performance monitoring"""
    
    def __init__(self):
        super().__init__("example_tool")
    
    def process_data(self, data: dict) -> dict:
        """Process data with monitoring"""
        return self.execute_with_monitoring("process_data", self._do_process, data)
    
    def _do_process(self, data: dict) -> dict:
        # Simulate data processing
        time.sleep(0.15)
        return {"processed": True, "input_size": len(str(data))}

# Alternative using decorators
@monitor_tool_execution("data_processor", "transform")
def transform_data(data: list) -> list:
    time.sleep(0.05)
    return [item.upper() for item in data]
'''
    
    # Save example files
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    with open(os.path.join(examples_dir, 'monitored_agent_example.py'), 'w') as f:
        f.write(example_agent_code)
    
    with open(os.path.join(examples_dir, 'monitored_tool_example.py'), 'w') as f:
        f.write(example_tool_code)
    
    logging.info("Example integration files created in examples/ directory")


def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Setting up AgentKen Performance Monitoring System")
    print("=" * 60)
    
    # Initialize setup
    setup = PerformanceMonitoringSetup()
    
    # Initialize monitoring
    print("📊 Initializing performance monitoring...")
    monitor = setup.initialize_monitoring()
    
    if monitor:
        print("✅ Performance monitoring initialized successfully")
    else:
        print("❌ Performance monitoring initialization failed")
        return
    
    # Setup dashboard
    print("🖥️  Setting up performance dashboard...")
    dashboard = setup.setup_dashboard()
    
    if dashboard:
        print("✅ Dashboard setup complete")
    else:
        print("❌ Dashboard setup failed")
    
    # Integrate with existing components
    print("🔗 Integrating with existing agents and tools...")
    setup.integrate_with_agents()
    setup.integrate_with_tools()
    
    # Create example integrations
    print("📝 Creating example integration files...")
    create_example_integration()
    
    # Generate setup report
    report = setup.generate_setup_report()
    
    print("\n📋 Setup Report:")
    print(f"  Monitoring Enabled: {report['monitoring_enabled']}")
    print(f"  Dashboard Enabled: {report['dashboard_enabled']}")
    print(f"  Database Path: {report['database_path']}")
    print(f"  Alert Rules: {report['alert_rules_count']}")
    if report['dashboard_url']:
        print(f"  Dashboard URL: {report['dashboard_url']}")
    
    print("\n🎉 Performance monitoring setup complete!")
    print("\nNext steps:")
    print("1. Run the dashboard: python setup_performance_monitoring.py --dashboard")
    print("2. Check examples/ directory for integration examples")
    print("3. Modify performance_config.json to customize settings")
    print("4. Integrate monitoring into your agents and tools")
    
    # Option to run dashboard immediately
    import sys
    if '--dashboard' in sys.argv:
        print("\n🖥️  Starting dashboard server...")
        setup.run_dashboard(debug='--debug' in sys.argv)


if __name__ == "__main__":
    main()