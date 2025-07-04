#!/usr/bin/env python3
"""
Client Example for Inferra V Enhanced System
Demonstrates how to interact with the system programmatically.
"""

import requests
import time
import json
from typing import Dict, Any, Optional, List


class InferraVClient:
    """Client for interacting with Inferra V API"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "InferraV-Client/1.0"}
        )

    def _request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        url = f"{self.base_url}/api{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def initialize_system(self) -> Dict[str, Any]:
        """Initialize the Inferra V system"""
        return self._request("POST", "/initialize")

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self._request("GET", "/status")

    def create_plan(self, name: str, description: str, tasks: list) -> str:
        """Create a new execution plan"""
        data = {"name": name, "description": description, "tasks": tasks}
        response = self._request("POST", "/plans", data)
        return response["data"]["plan_id"]

    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a plan"""
        return self._request("POST", f"/plans/{plan_id}/execute")

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get plan execution status"""
        return self._request("GET", f"/plans/{plan_id}/status")

    def monitor_plan(self, plan_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Monitor plan execution until completion"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status_response = self.get_plan_status(plan_id)
            status = status_response["data"]

            print(
                f"📊 Progress: {status['progress']:.1%} ({status['completed_tasks']}/{status['total_tasks']} tasks)"
            )

            if status["status"] in ["completed", "failed"]:
                return status

            time.sleep(2)

        raise Exception(f"Plan monitoring timed out after {timeout} seconds")

    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans"""
        response = self._request("GET", "/plans")
        return response["data"]["plans"]

    def get_optimization(self) -> Dict[str, Any]:
        """Get optimization recommendations"""
        return self._request("GET", "/optimize")

    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Chat with the system"""
        data = {"message": message}
        if session_id:
            data["session_id"] = session_id
        return self._request("POST", "/chat", data)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return self._request("GET", "/metrics")

    def shutdown_system(self) -> Dict[str, Any]:
        """Shutdown the system"""
        return self._request("POST", "/shutdown")


def demo_basic_usage():
    """Demonstrate basic usage of the Inferra V client"""
    print("🚀 Inferra V Client Demo")
    print("=" * 40)

    client = InferraVClient()

    try:
        # 1. Initialize system
        print("1️⃣ Initializing system...")
        init_response = client.initialize_system()
        print(f"✅ {init_response['data']['message']}")

        # 2. Check status
        print("\n2️⃣ Checking system status...")
        status = client.get_status()["data"]
        print(f"📊 Health Score: {status['health_score']:.1f}/100")
        print(f"🤖 Agents: {status['agents']}")

        # 3. Create a plan
        print("\n3️⃣ Creating execution plan...")
        tasks = [
            {
                "id": "demo_task_1",
                "name": "Web Search Demo",
                "agent_type": "research_agent",
                "capability": "web_search",
                "parameters": {"query": "AI trends 2024", "max_results": 3},
                "priority": 3,
                "estimated_duration": 5.0,
                "dependencies": [],
                "required_resources": ["internet_access"],
            },
            {
                "id": "demo_task_2",
                "name": "Data Processing Demo",
                "agent_type": "research_agent",
                "capability": "data_processing",
                "parameters": {"data": {"demo": "data"}, "operation": "analyze"},
                "priority": 2,
                "estimated_duration": 3.0,
                "dependencies": ["demo_task_1"],
                "required_resources": ["cpu_intensive"],
            },
        ]

        plan_id = client.create_plan(
            name="Client Demo Plan",
            description="Demonstration plan created via API client",
            tasks=tasks,
        )
        print(f"✅ Plan created: {plan_id}")

        # 4. Execute plan
        print("\n4️⃣ Executing plan...")
        client.execute_plan(plan_id)
        print("🚀 Plan execution started")

        # 5. Monitor execution
        print("\n5️⃣ Monitoring execution...")
        final_status = client.monitor_plan(plan_id, timeout=30)
        print(f"✅ Plan {final_status['status']}!")

        # 6. Get optimization recommendations
        print("\n6️⃣ Getting optimization recommendations...")
        optimization = client.get_optimization()["data"]
        print(f"💡 Recommendations: {len(optimization['recommendations'])}")

        if optimization["recommendations"]:
            top_rec = optimization["recommendations"][0]
            print(f"🎯 Top recommendation: {top_rec['title']}")
            print(f"   Expected improvement: {top_rec['expected_improvement']:.1f}%")

        # 7. Chat with system
        print("\n7️⃣ Chatting with system...")
        chat_response = client.chat("What is the current system status?")
        print(f"🤖 Response: {chat_response['data']['response']}")

        print("\n🎉 Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


def demo_advanced_usage():
    """Demonstrate advanced usage patterns"""
    print("\n🔧 Advanced Usage Demo")
    print("=" * 40)

    client = InferraVClient()

    try:
        # Ensure system is initialized
        client.initialize_system()

        # Create multiple plans
        print("📋 Creating multiple plans...")
        plan_ids = []

        for i in range(3):
            tasks = [
                {
                    "id": f"batch_task_{i}",
                    "name": f"Batch Task {i+1}",
                    "agent_type": "research_agent",
                    "capability": "web_search",
                    "parameters": {"query": f"batch query {i+1}", "max_results": 2},
                    "priority": 2,
                    "estimated_duration": 2.0,
                    "dependencies": [],
                    "required_resources": ["internet_access"],
                }
            ]

            plan_id = client.create_plan(
                name=f"Batch Plan {i+1}",
                description=f"Batch processing plan {i+1}",
                tasks=tasks,
            )
            plan_ids.append(plan_id)

        print(f"✅ Created {len(plan_ids)} plans")

        # Execute all plans
        print("🚀 Executing all plans...")
        for plan_id in plan_ids:
            client.execute_plan(plan_id)

        # Monitor all plans
        print("📊 Monitoring all plans...")
        completed_plans = 0

        while completed_plans < len(plan_ids):
            completed_plans = 0

            for plan_id in plan_ids:
                status = client.get_plan_status(plan_id)["data"]
                if status["status"] in ["completed", "failed"]:
                    completed_plans += 1

            print(f"Progress: {completed_plans}/{len(plan_ids)} plans completed")

            if completed_plans < len(plan_ids):
                time.sleep(2)

        print("✅ All plans completed!")

        # Get final metrics
        metrics = client.get_metrics()["data"]
        print(f"📈 Final health score: {metrics['health_score']:.1f}/100")

    except Exception as e:
        print(f"❌ Advanced demo failed: {e}")


def interactive_client():
    """Interactive client for manual testing"""
    print("\n💬 Interactive Client Mode")
    print("=" * 40)
    print("Commands: status, create, execute, chat, optimize, quit")

    client = InferraVClient()

    # Initialize system
    try:
        client.initialize_system()
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    while True:
        try:
            command = input("\n🤖 Command: ").strip().lower()

            if command == "quit":
                break
            elif command == "status":
                status = client.get_status()["data"]
                print(
                    f"Health: {status['health_score']:.1f}/100, Plans: {status['total_plans']}"
                )
            elif command == "create":
                # Simple plan creation
                plan_id = client.create_plan(
                    name="Interactive Plan",
                    description="Plan created interactively",
                    tasks=[
                        {
                            "id": "interactive_task",
                            "name": "Interactive Task",
                            "agent_type": "research_agent",
                            "capability": "web_search",
                            "parameters": {
                                "query": "interactive test",
                                "max_results": 2,
                            },
                            "priority": 2,
                            "estimated_duration": 3.0,
                            "dependencies": [],
                            "required_resources": ["internet_access"],
                        }
                    ],
                )
                print(f"✅ Plan created: {plan_id}")
            elif command == "execute":
                plans = client.list_plans()
                if plans:
                    plan_id = plans[-1]["id"]  # Execute latest plan
                    client.execute_plan(plan_id)
                    print(f"🚀 Executing plan: {plan_id}")
                else:
                    print("❌ No plans available")
            elif command == "chat":
                message = input("Message: ")
                response = client.chat(message)
                print(f"🤖 {response['data']['response']}")
            elif command == "optimize":
                opt = client.get_optimization()["data"]
                print(f"💡 {len(opt['recommendations'])} recommendations available")
            else:
                print("❓ Unknown command")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("👋 Goodbye!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "advanced":
            demo_advanced_usage()
        elif mode == "interactive":
            interactive_client()
        else:
            print("Usage: python client_example.py [advanced|interactive]")
    else:
        demo_basic_usage()
