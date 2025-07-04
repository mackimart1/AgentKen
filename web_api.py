#!/usr/bin/env python3
"""
Web API Interface for Inferra V Enhanced System
Provides RESTful API endpoints for external applications to interact with the agent system.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not installed. Install with: pip install flask flask-cors")

from core.workflow_monitoring import WorkflowMonitor
from core.tool_integration_system import create_tool_system
from core.agent_framework import create_agent_system
from core.adaptive_orchestrator import (
    AdvancedOrchestrator,
    TaskPriority,
    AgentCapacity,
    ResourceConstraint,
)


class InferraVAPI:
    """RESTful API for Inferra V Enhanced System"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web applications

        # System components
        self.monitor = None
        self.orchestrator = None
        self.agents = []
        self.system_initialized = False

        # API state
        self.active_sessions = {}
        self.api_stats = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "start_time": time.time(),
        }

        self._setup_routes()

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.route("/")
        def home():
            """API documentation and status page"""
            return render_template_string(self._get_home_template())

        @self.app.route("/api/status", methods=["GET"])
        def get_status():
            """Get system status"""
            return self._handle_request(self._get_system_status)

        @self.app.route("/api/initialize", methods=["POST"])
        def initialize_system():
            """Initialize the Inferra V system"""
            return self._handle_request(self._initialize_system)

        @self.app.route("/api/plans", methods=["GET"])
        def list_plans():
            """List all plans"""
            return self._handle_request(self._list_plans)

        @self.app.route("/api/plans", methods=["POST"])
        def create_plan():
            """Create a new plan"""
            return self._handle_request(self._create_plan, request.get_json())

        @self.app.route("/api/plans/<plan_id>", methods=["GET"])
        def get_plan(plan_id):
            """Get plan details"""
            return self._handle_request(self._get_plan, plan_id)

        @self.app.route("/api/plans/<plan_id>/execute", methods=["POST"])
        def execute_plan(plan_id):
            """Execute a plan"""
            return self._handle_request(self._execute_plan, plan_id)

        @self.app.route("/api/plans/<plan_id>/status", methods=["GET"])
        def get_plan_status(plan_id):
            """Get plan execution status"""
            return self._handle_request(self._get_plan_status, plan_id)

        @self.app.route("/api/agents", methods=["GET"])
        def list_agents():
            """List all agents"""
            return self._handle_request(self._list_agents)

        @self.app.route("/api/tools", methods=["GET"])
        def list_tools():
            """List all tools"""
            return self._handle_request(self._list_tools)

        @self.app.route("/api/optimize", methods=["GET"])
        def get_optimization():
            """Get optimization recommendations"""
            return self._handle_request(self._get_optimization)

        @self.app.route("/api/chat", methods=["POST"])
        def chat():
            """Chat with the system"""
            return self._handle_request(self._chat, request.get_json())

        @self.app.route("/api/metrics", methods=["GET"])
        def get_metrics():
            """Get system metrics"""
            return self._handle_request(self._get_metrics)

        @self.app.route("/api/alerts", methods=["GET"])
        def get_alerts():
            """Get active alerts"""
            return self._handle_request(self._get_alerts)

        @self.app.route("/api/shutdown", methods=["POST"])
        def shutdown_system():
            """Shutdown the system"""
            return self._handle_request(self._shutdown_system)

    def _handle_request(self, handler, *args):
        """Handle API requests with error handling and stats"""
        self.api_stats["requests_total"] += 1

        try:
            result = handler(*args)
            self.api_stats["requests_successful"] += 1
            return jsonify({"success": True, "data": result, "timestamp": time.time()})
        except Exception as e:
            self.api_stats["requests_failed"] += 1
            self.logger.error(f"API request failed: {e}")
            return (
                jsonify({"success": False, "error": str(e), "timestamp": time.time()}),
                500,
            )

    def _get_system_status(self):
        """Get comprehensive system status"""
        if not self.system_initialized:
            return {
                "initialized": False,
                "message": "System not initialized. Call /api/initialize first.",
            }

        health_score = self.monitor.get_system_health_score()
        metrics = self.orchestrator.get_system_metrics()
        active_alerts = self.monitor.alert_manager.get_active_alerts()

        return {
            "initialized": True,
            "health_score": health_score,
            "active_plans": metrics["active_plans"],
            "total_plans": metrics["total_plans"],
            "agents": len(self.agents),
            "active_alerts": len(active_alerts),
            "uptime": time.time() - self.api_stats["start_time"],
            "api_stats": self.api_stats,
        }

    def _initialize_system(self):
        """Initialize the Inferra V system"""
        if self.system_initialized:
            return {"message": "System already initialized"}

        # Initialize monitoring
        self.monitor = WorkflowMonitor()
        self.monitor.start()

        # Initialize tools and agents
        tool_registry = create_tool_system()
        message_bus, _, self.agents = create_agent_system()

        # Initialize orchestrator
        self.orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

        # Configure system
        self.orchestrator.agent_capacities["research_agent"] = AgentCapacity(
            agent_id="research_agent",
            max_concurrent=3,
            capabilities=["web_search", "data_processing"],
            performance_score=0.9,
        )

        self.orchestrator.resource_constraints.update(
            {
                "internet_access": ResourceConstraint("internet_access", 10),
                "cpu_intensive": ResourceConstraint("cpu_intensive", 5),
                "storage": ResourceConstraint("storage", 100),
                "network_bandwidth": ResourceConstraint("network_bandwidth", 20),
            }
        )

        self.system_initialized = True

        return {
            "message": "System initialized successfully",
            "agents": len(self.agents),
            "tools": len(tool_registry.tools),
        }

    def _list_plans(self):
        """List all plans"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        plans = []
        for plan_id, plan in self.orchestrator.active_plans.items():
            status = self.orchestrator.get_plan_status(plan_id)
            plans.append(
                {
                    "id": plan_id,
                    "name": plan.name,
                    "description": plan.description,
                    "status": status["status"] if status else "unknown",
                    "progress": status["progress"] if status else 0,
                    "total_tasks": len(plan.tasks),
                    "created_time": plan.created_time,
                }
            )

        return {"plans": plans}

    def _create_plan(self, data):
        """Create a new plan"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        if not data:
            raise Exception("No plan data provided")

        # Validate required fields
        required_fields = ["name", "description", "tasks"]
        for field in required_fields:
            if field not in data:
                raise Exception(f"Missing required field: {field}")

        # Create plan
        plan_id = self.orchestrator.create_plan(
            name=data["name"],
            description=data["description"],
            tasks=data["tasks"],
            deadline=data.get("deadline"),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
        )

        return {"plan_id": plan_id, "message": "Plan created successfully"}

    def _get_plan(self, plan_id):
        """Get plan details"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        if plan_id not in self.orchestrator.active_plans:
            raise Exception(f"Plan {plan_id} not found")

        plan = self.orchestrator.active_plans[plan_id]
        status = self.orchestrator.get_plan_status(plan_id)

        return {
            "id": plan_id,
            "name": plan.name,
            "description": plan.description,
            "status": status,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "capability": task.capability,
                    "status": task.status,
                    "priority": task.priority.name,
                    "estimated_duration": task.estimated_duration,
                }
                for task in plan.tasks
            ],
            "created_time": plan.created_time,
            "deadline": plan.deadline,
        }

    def _execute_plan(self, plan_id):
        """Execute a plan"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        success = self.orchestrator.execute_plan(plan_id)

        if not success:
            raise Exception("Failed to start plan execution")

        return {"message": "Plan execution started", "plan_id": plan_id}

    def _get_plan_status(self, plan_id):
        """Get plan execution status"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        status = self.orchestrator.get_plan_status(plan_id)

        if not status:
            raise Exception(f"Plan {plan_id} not found")

        return status

    def _list_agents(self):
        """List all agents"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        agents_info = []
        for agent in self.agents:
            agents_info.append(
                {
                    "id": agent.agent_id,
                    "status": agent.status.value,
                    "capabilities": [cap.name for cap in agent.capabilities.values()],
                    "active_tasks": len(agent.active_tasks),
                }
            )

        return {"agents": agents_info}

    def _list_tools(self):
        """List all tools"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        tools_info = []
        for name, tool in self.orchestrator.tool_registry.tools.items():
            status = tool.get_status()
            tools_info.append(
                {
                    "name": name,
                    "version": tool.definition.version,
                    "description": tool.definition.description,
                    "status": status["status"],
                    "performance_metrics": status["performance_metrics"],
                }
            )

        return {"tools": tools_info}

    def _get_optimization(self):
        """Get optimization recommendations"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        report = self.monitor.get_optimization_report()

        # Convert dataclass objects to dictionaries
        recommendations = []
        for rec in report["optimization_recommendations"]:
            recommendations.append(
                {
                    "id": rec.id,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_effort": rec.implementation_effort,
                    "risk_level": rec.risk_level,
                    "priority_score": rec.priority_score,
                    "affected_components": rec.affected_components,
                }
            )

        return {
            "system_health_score": report["system_health_score"],
            "active_alerts_count": report["active_alerts_count"],
            "bottlenecks_detected": report["bottlenecks_detected"],
            "recommendations": recommendations,
        }

    def _chat(self, data):
        """Chat with the system"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        if not data or "message" not in data:
            raise Exception("No message provided")

        user_message = data["message"]
        session_id = data.get("session_id", str(uuid.uuid4()))

        # Simple response generation (can be enhanced with LLM)
        response = self._generate_chat_response(user_message)

        # Store conversation
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []

        self.active_sessions[session_id].append(
            {"user": user_message, "agent": response, "timestamp": time.time()}
        )

        return {"response": response, "session_id": session_id}

    def _generate_chat_response(self, message):
        """Generate chat response"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["status", "health"]):
            health = self.monitor.get_system_health_score()
            return f"System health is {health:.1f}/100. All systems operational!"

        elif any(word in message_lower for word in ["plan", "create"]):
            return "I can help you create and execute plans! Use the /api/plans endpoint to create new plans."

        elif any(word in message_lower for word in ["help", "what"]):
            return "I'm Inferra V API! I can manage plans, execute tasks, monitor performance, and optimize workflows. Check the API documentation for available endpoints."

        else:
            return "I understand you want to interact with the system. Try asking about status, plans, or optimization!"

    def _get_metrics(self):
        """Get system metrics"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        metrics = self.orchestrator.get_system_metrics()
        health_score = self.monitor.get_system_health_score()

        return {
            "health_score": health_score,
            "system_metrics": metrics,
            "api_stats": self.api_stats,
        }

    def _get_alerts(self):
        """Get active alerts"""
        if not self.system_initialized:
            raise Exception("System not initialized")

        active_alerts = self.monitor.alert_manager.get_active_alerts()

        alerts_data = []
        for alert in active_alerts:
            alerts_data.append(
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                    "acknowledged": alert.acknowledged,
                }
            )

        return {"alerts": alerts_data}

    def _shutdown_system(self):
        """Shutdown the system"""
        if not self.system_initialized:
            return {"message": "System not initialized"}

        # Shutdown components
        if self.monitor:
            self.monitor.stop()

        for agent in self.agents:
            agent.shutdown()

        self.system_initialized = False

        return {"message": "System shutdown complete"}

    def _get_home_template(self):
        """Get HTML template for API documentation"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Inferra V Enhanced API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
        .method { font-weight: bold; color: #e74c3c; }
        .path { font-family: monospace; background: #2c3e50; color: white; padding: 2px 6px; border-radius: 3px; }
        .description { margin-top: 5px; color: #7f8c8d; }
        .status { padding: 10px; background: #d5f4e6; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Inferra V Enhanced API</h1>
        
        <div class="status">
            <strong>API Status:</strong> Online and Ready<br>
            <strong>Version:</strong> 1.0.0<br>
            <strong>Documentation:</strong> Interactive API for Inferra V Enhanced System
        </div>
        
        <h2>📋 Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/status</span>
            <div class="description">Get system status and health information</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/api/initialize</span>
            <div class="description">Initialize the Inferra V system</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/plans</span>
            <div class="description">List all execution plans</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/api/plans</span>
            <div class="description">Create a new execution plan</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/plans/{id}</span>
            <div class="description">Get details of a specific plan</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/api/plans/{id}/execute</span>
            <div class="description">Execute a specific plan</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/plans/{id}/status</span>
            <div class="description">Get execution status of a plan</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/agents</span>
            <div class="description">List all available agents</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/tools</span>
            <div class="description">List all available tools</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/optimize</span>
            <div class="description">Get optimization recommendations</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/api/chat</span>
            <div class="description">Chat with the system</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/metrics</span>
            <div class="description">Get system performance metrics</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/alerts</span>
            <div class="description">Get active system alerts</div>
        </div>
        
        <h2>🔧 Usage Examples</h2>
        
        <h3>Initialize System</h3>
        <pre>curl -X POST http://localhost:5000/api/initialize</pre>
        
        <h3>Get System Status</h3>
        <pre>curl http://localhost:5000/api/status</pre>
        
        <h3>Create a Plan</h3>
        <pre>curl -X POST http://localhost:5000/api/plans \\
  -H "Content-Type: application/json" \\
  -d '{"name": "Test Plan", "description": "Test", "tasks": []}'</pre>
        
        <h3>Chat with System</h3>
        <pre>curl -X POST http://localhost:5000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is the system status?"}'</pre>
        
        <p><strong>Note:</strong> Make sure to initialize the system first before using other endpoints!</p>
    </div>
</body>
</html>
        """

    def run(self, host="localhost", port=5000, debug=False):
        """Run the API server"""
        if not FLASK_AVAILABLE:
            raise Exception(
                "Flask is required to run the web API. Install with: pip install flask flask-cors"
            )

        print(f"🚀 Starting Inferra V API server on http://{host}:{port}")
        print("📖 Visit the URL above for API documentation")

        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to start the API server"""
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Install with: pip install flask flask-cors")
        return

    api = InferraVAPI()

    try:
        api.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ API server error: {e}")


if __name__ == "__main__":
    main()
