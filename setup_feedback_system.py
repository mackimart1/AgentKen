"""
Setup script for integrating the User Feedback Loop System into AgentKen.
This script configures and initializes the feedback collection and learning system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from feedback_system import FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, ContinuousLearningEngine
from feedback_integration import initialize_feedback_system, FeedbackSystem
from feedback_interface import FeedbackWebInterface, create_feedback_templates


class FeedbackSystemSetup:
    """Setup and configuration for the feedback system"""
    
    def __init__(self, config_path: str = "feedback_config.json"):
        self.config_path = config_path
        self.config = self._load_or_create_config()
        self.feedback_system = None
        self.web_interface = None
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load configuration or create default"""
        default_config = {
            "database": {
                "path": "feedback_system.db",
                "retention_days": 365
            },
            "collection": {
                "enabled": True,
                "auto_prompt": True,
                "prompt_strategy": "smart",
                "feedback_rate_target": 0.3
            },
            "learning": {
                "enabled": True,
                "auto_apply": True,
                "confidence_threshold": 0.7,
                "impact_threshold": 5.0
            },
            "web_interface": {
                "enabled": True,
                "host": "localhost",
                "port": 5002,
                "auto_refresh_seconds": 30
            },
            "prompting": {
                "strategies": {
                    "immediate": {
                        "enabled": True,
                        "conditions": ["failure", "error"]
                    },
                    "delayed": {
                        "enabled": True,
                        "delay_seconds": 300,
                        "conditions": ["long_execution"]
                    },
                    "smart": {
                        "enabled": True,
                        "min_execution_time": 1.0,
                        "failure_always_prompt": True,
                        "success_prompt_rate": 0.3
                    }
                }
            },
            "notifications": {
                "console": True,
                "web": True,
                "email": False,
                "webhook": False
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
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
    
    def initialize_feedback_system(self) -> FeedbackSystem:
        """Initialize the feedback system"""
        if not self.config["collection"]["enabled"]:
            logging.info("Feedback collection is disabled in configuration")
            return None
        
        db_path = self.config["database"]["path"]
        self.feedback_system = initialize_feedback_system(db_path)
        
        # Configure prompting
        self.feedback_system.set_prompt_strategy(self.config["collection"]["prompt_strategy"])
        self.feedback_system.enable_auto_prompt(self.config["collection"]["auto_prompt"])
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        # Register additional learning strategies
        self._register_custom_learning_strategies()
        
        logging.info("Feedback system initialized successfully")
        return self.feedback_system
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        if self.config["notifications"]["console"]:
            from feedback_integration import console_feedback_prompt_handler
            self.feedback_system.add_feedback_handler(console_feedback_prompt_handler)
        
        if self.config["notifications"]["web"]:
            from feedback_integration import web_feedback_prompt_handler
            self.feedback_system.add_feedback_handler(web_feedback_prompt_handler)
        
        # Add custom handlers here if needed
        if self.config["notifications"]["email"]:
            logging.warning("Email notifications not implemented yet")
        
        if self.config["notifications"]["webhook"]:
            logging.warning("Webhook notifications not implemented yet")
    
    def _register_custom_learning_strategies(self):
        """Register custom learning strategies"""
        
        def adaptive_timeout_strategy(insight):
            """Adjust timeouts based on user feedback about speed"""
            if insight.insight_type == "performance_correlation":
                correlation = insight.supporting_data.get("correlation", 0)
                if correlation < -0.6:  # Strong negative correlation with execution time
                    return {
                        "type": "timeout_adjustment",
                        "component_id": insight.component_id,
                        "operation": insight.operation,
                        "adjustments": {
                            "timeout_multiplier": 0.8,
                            "async_processing": True,
                            "caching_enabled": True
                        },
                        "reason": "Reduce execution time based on user feedback"
                    }
            return None
        
        def user_preference_strategy(insight):
            """Adapt based on user preference patterns"""
            if insight.insight_type == "satisfaction_variability":
                return {
                    "type": "preference_adaptation",
                    "component_id": insight.component_id,
                    "operation": insight.operation,
                    "adjustments": {
                        "personalization_enabled": True,
                        "user_preference_tracking": True,
                        "adaptive_responses": True
                    },
                    "reason": "Adapt to user preferences to reduce satisfaction variability"
                }
            return None
        
        self.feedback_system.learning_engine.register_learning_strategy(
            "adaptive_timeout", adaptive_timeout_strategy
        )
        self.feedback_system.learning_engine.register_learning_strategy(
            "user_preference", user_preference_strategy
        )
    
    def setup_web_interface(self) -> FeedbackWebInterface:
        """Setup the feedback web interface"""
        if not self.config["web_interface"]["enabled"] or not self.feedback_system:
            logging.info("Web interface is disabled or feedback system not initialized")
            return None
        
        # Create templates
        self._create_web_templates()
        
        # Create web interface
        self.web_interface = FeedbackWebInterface(
            feedback_collector=self.feedback_system.collector,
            feedback_analyzer=self.feedback_system.analyzer,
            learning_engine=self.feedback_system.learning_engine,
            host=self.config["web_interface"]["host"],
            port=self.config["web_interface"]["port"]
        )
        
        logging.info(f"Web interface setup complete. Access at http://{self.config['web_interface']['host']}:{self.config['web_interface']['port']}")
        return self.web_interface
    
    def _create_web_templates(self):
        """Create web templates for the feedback interface"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        dashboard_template, feedback_form_template = create_feedback_templates()
        
        # Save templates
        with open(os.path.join(templates_dir, 'feedback_dashboard.html'), 'w') as f:
            f.write(dashboard_template)
        
        with open(os.path.join(templates_dir, 'feedback_form.html'), 'w') as f:
            f.write(feedback_form_template)
        
        logging.info("Web templates created successfully")
    
    def integrate_with_agents(self, agents_manifest_path: str = "agents_manifest.json"):
        """Integrate feedback collection with existing agents"""
        if not self.config["collection"]["enabled"]:
            return
        
        try:
            if os.path.exists(agents_manifest_path):
                with open(agents_manifest_path, 'r') as f:
                    agents_manifest = json.load(f)
                
                # Handle both dict and list formats
                if isinstance(agents_manifest, list):
                    agents_dict = {agent.get('id', f'agent_{i}'): agent for i, agent in enumerate(agents_manifest)}
                else:
                    agents_dict = agents_manifest
                
                for agent_id, agent_info in agents_dict.items():
                    logging.info(f"Feedback integration available for agent: {agent_id}")
                    # Integration would happen here in a real implementation
        except Exception as e:
            logging.error(f"Failed to integrate with agents: {e}")
    
    def integrate_with_tools(self, tools_manifest_path: str = "tools_manifest.json"):
        """Integrate feedback collection with existing tools"""
        if not self.config["collection"]["enabled"]:
            return
        
        try:
            if os.path.exists(tools_manifest_path):
                with open(tools_manifest_path, 'r') as f:
                    tools_manifest = json.load(f)
                
                # Handle both dict and list formats
                if isinstance(tools_manifest, list):
                    tools_dict = {tool.get('id', f'tool_{i}'): tool for i, tool in enumerate(tools_manifest)}
                else:
                    tools_dict = tools_manifest
                
                for tool_id, tool_info in tools_dict.items():
                    logging.info(f"Feedback integration available for tool: {tool_id}")
                    # Integration would happen here in a real implementation
        except Exception as e:
            logging.error(f"Failed to integrate with tools: {e}")
    
    def run_web_interface(self, debug: bool = False):
        """Run the web interface"""
        if self.web_interface:
            self.web_interface.run(debug=debug)
        else:
            logging.error("Web interface not initialized")
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate a setup report"""
        return {
            "feedback_system_enabled": self.feedback_system is not None,
            "web_interface_enabled": self.web_interface is not None,
            "config_path": self.config_path,
            "database_path": self.config["database"]["path"],
            "auto_prompt_enabled": self.config["collection"]["auto_prompt"],
            "learning_enabled": self.config["learning"]["enabled"],
            "web_interface_url": f"http://{self.config['web_interface']['host']}:{self.config['web_interface']['port']}" if self.web_interface else None,
            "configuration": self.config
        }


def create_integration_examples():
    """Create example integration files"""
    
    # Example agent with feedback integration
    agent_example = '''
from core.feedback_integration import FeedbackEnabledAgent, track_agent_execution
import time

class ExampleFeedbackAgent(FeedbackEnabledAgent):
    """Example agent with built-in feedback collection"""
    
    def __init__(self):
        super().__init__("example_feedback_agent")
    
    def research_topic(self, topic: str, user_id: str = None, session_id: str = None) -> dict:
        """Research a topic with feedback tracking"""
        return self.execute_with_feedback(
            operation="research_topic",
            func=self._do_research,
            user_id=user_id,
            session_id=session_id,
            input_data={"topic": topic},
            topic=topic
        )
    
    def _do_research(self, topic: str) -> dict:
        # Simulate research work
        time.sleep(0.2)
        return {
            "topic": topic,
            "results": ["result1", "result2", "result3"],
            "confidence": 0.85
        }

# Alternative using decorators
@track_agent_execution("research_agent", "web_search")
def search_web(query: str, user_id: str = None, session_id: str = None) -> dict:
    time.sleep(0.1)
    return {
        "query": query,
        "results": ["web result 1", "web result 2"],
        "search_time": 0.1
    }

# Usage example
if __name__ == "__main__":
    from core.feedback_integration import initialize_feedback_system
    
    # Initialize feedback system
    feedback_system = initialize_feedback_system()
    
    # Create agent
    agent = ExampleFeedbackAgent()
    
    # Execute with feedback tracking
    result = agent.research_topic("AI trends", user_id="user123", session_id="session456")
    print(f"Research result: {result}")
    
    # Get performance profile
    profile = agent.get_performance_profile()
    if profile:
        print(f"Agent performance: {profile.average_rating:.2f}/5.0")
'''
    
    # Example tool with feedback integration
    tool_example = '''
from core.feedback_integration import FeedbackEnabledTool, track_tool_execution
import time

class ExampleFeedbackTool(FeedbackEnabledTool):
    """Example tool with built-in feedback collection"""
    
    def __init__(self):
        super().__init__("example_feedback_tool")
    
    def process_data(self, data: dict, user_id: str = None, session_id: str = None) -> dict:
        """Process data with feedback tracking"""
        return self.execute_with_feedback(
            operation="process_data",
            func=self._do_process,
            user_id=user_id,
            session_id=session_id,
            input_data=data,
            data=data
        )
    
    def _do_process(self, data: dict) -> dict:
        # Simulate data processing
        time.sleep(0.15)
        return {
            "processed": True,
            "input_size": len(str(data)),
            "output_quality": 0.9
        }

# Alternative using decorators
@track_tool_execution("data_processor", "transform")
def transform_data(data: list, user_id: str = None, session_id: str = None) -> list:
    time.sleep(0.05)
    return [item.upper() for item in data]

# Usage example
if __name__ == "__main__":
    from core.feedback_integration import initialize_feedback_system, collect_rating_feedback
    
    # Initialize feedback system
    feedback_system = initialize_feedback_system()
    
    # Create tool
    tool = ExampleFeedbackTool()
    
    # Execute with feedback tracking
    result = tool.process_data({"test": "data"}, user_id="user123", session_id="session456")
    print(f"Processing result: {result}")
    
    # Manually collect feedback
    recent_tasks = feedback_system.collector.get_pending_tasks_for_user("user123")
    if recent_tasks:
        feedback = collect_rating_feedback(
            task_execution_id=recent_tasks[0].id,
            user_id="user123",
            rating=4.5,
            text_feedback="Great tool, very helpful!"
        )
        print(f"Feedback collected: {feedback.id}")
'''
    
    # Save example files
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    with open(os.path.join(examples_dir, 'feedback_agent_example.py'), 'w') as f:
        f.write(agent_example)
    
    with open(os.path.join(examples_dir, 'feedback_tool_example.py'), 'w') as f:
        f.write(tool_example)
    
    logging.info("Example integration files created in examples/ directory")


def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔄 Setting up AgentKen User Feedback Loop System")
    print("=" * 60)
    
    # Initialize setup
    setup = FeedbackSystemSetup()
    
    # Initialize feedback system
    print("📊 Initializing feedback system...")
    feedback_system = setup.initialize_feedback_system()
    
    if feedback_system:
        print("✅ Feedback system initialized successfully")
    else:
        print("❌ Feedback system initialization failed")
        return
    
    # Setup web interface
    print("🖥️  Setting up feedback web interface...")
    web_interface = setup.setup_web_interface()
    
    if web_interface:
        print("✅ Web interface setup complete")
    else:
        print("❌ Web interface setup failed")
    
    # Integrate with existing components
    print("🔗 Integrating with existing agents and tools...")
    setup.integrate_with_agents()
    setup.integrate_with_tools()
    
    # Create example integrations
    print("📝 Creating example integration files...")
    create_integration_examples()
    
    # Generate setup report
    report = setup.generate_setup_report()
    
    print("\n📋 Setup Report:")
    print(f"  Feedback System Enabled: {report['feedback_system_enabled']}")
    print(f"  Web Interface Enabled: {report['web_interface_enabled']}")
    print(f"  Database Path: {report['database_path']}")
    print(f"  Auto Prompt Enabled: {report['auto_prompt_enabled']}")
    print(f"  Learning Enabled: {report['learning_enabled']}")
    if report['web_interface_url']:
        print(f"  Web Interface URL: {report['web_interface_url']}")
    
    print("\n🎉 Feedback system setup complete!")
    print("\nNext steps:")
    print("1. Run the web interface: python setup_feedback_system.py --web")
    print("2. Run the test suite: python test_feedback_system.py")
    print("3. Check examples/ directory for integration examples")
    print("4. Modify feedback_config.json to customize settings")
    print("5. Integrate feedback tracking into your agents and tools")
    
    # Option to run web interface immediately
    import sys
    if '--web' in sys.argv:
        print("\n🖥️  Starting web interface...")
        setup.run_web_interface(debug='--debug' in sys.argv)


if __name__ == "__main__":
    main()