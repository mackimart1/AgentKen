"""
Enhanced Agent Communication and Coordination Framework
Implements modern design patterns for multi-agent systems with improved reliability and performance.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, Future
import threading


class MessageType(Enum):
    """Types of messages in the agent communication system"""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"


class AgentStatus(Enum):
    """Agent operational status"""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class Message:
    """Standard message format for agent communication"""

    id: str
    type: MessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1=low, 5=high
    timeout: Optional[float] = None
    correlation_id: Optional[str] = None


@dataclass
class AgentCapability:
    """Describes what an agent can do"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    estimated_duration: Optional[float] = None
    max_concurrent: int = 1
    dependencies: List[str] = None


@dataclass
class TaskResult:
    """Result of a task execution"""

    task_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None


class MessageBus:
    """Central message routing and delivery system"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.delivery_stats = {"sent": 0, "delivered": 0, "failed": 0}
        self._lock = threading.Lock()

    def subscribe(self, message_type: MessageType, handler: Callable[[Message], None]):
        """Subscribe to specific message types"""
        with self._lock:
            if message_type.value not in self.subscribers:
                self.subscribers[message_type.value] = []
            self.subscribers[message_type.value].append(handler)

    def publish(self, message: Message) -> bool:
        """Publish message to all subscribers"""
        try:
            with self._lock:
                self.message_history.append(message)
                self.delivery_stats["sent"] += 1

                subscribers = self.subscribers.get(message.type.value, [])
                for handler in subscribers:
                    try:
                        handler(message)
                        self.delivery_stats["delivered"] += 1
                    except Exception as e:
                        logging.error(f"Message delivery failed: {e}")
                        self.delivery_stats["failed"] += 1

            return True
        except Exception as e:
            logging.error(f"Message publish failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get message delivery statistics"""
        with self._lock:
            return {
                "delivery_stats": self.delivery_stats.copy(),
                "total_messages": len(self.message_history),
                "active_subscribers": {k: len(v) for k, v in self.subscribers.items()},
            }


class BaseAgent(ABC):
    """Base class for all agents with enhanced capabilities"""

    def __init__(
        self,
        agent_id: str,
        capabilities: List[AgentCapability],
        message_bus: MessageBus,
    ):
        self.agent_id = agent_id
        self.capabilities = {cap.name: cap for cap in capabilities}
        self.message_bus = message_bus
        self.status = AgentStatus.STARTING
        self.active_tasks: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.heartbeat_interval = 30.0
        self.last_heartbeat = time.time()

        # Subscribe to relevant messages
        self.message_bus.subscribe(MessageType.TASK_REQUEST, self._handle_task_request)
        self.message_bus.subscribe(
            MessageType.CAPABILITY_QUERY, self._handle_capability_query
        )
        self.message_bus.subscribe(MessageType.HEARTBEAT, self._handle_heartbeat)

        self.status = AgentStatus.IDLE
        logging.info(
            f"Agent {self.agent_id} initialized with {len(capabilities)} capabilities"
        )

    def _handle_task_request(self, message: Message):
        """Handle incoming task requests"""
        if message.recipient != self.agent_id:
            return

        payload = message.payload
        task_type = payload.get("task_type")

        if task_type not in self.capabilities:
            self._send_error_response(
                message, f"Capability '{task_type}' not supported"
            )
            return

        # Check concurrent task limits
        capability = self.capabilities[task_type]
        current_tasks = sum(
            1 for task in self.active_tasks.values() if task.get("type") == task_type
        )

        if current_tasks >= capability.max_concurrent:
            self._send_error_response(
                message, f"Max concurrent tasks ({capability.max_concurrent}) reached"
            )
            return

        # Execute task asynchronously
        task_id = str(uuid.uuid4())
        future = self.executor.submit(self._execute_task, task_id, task_type, payload)

        self.active_tasks[task_id] = {
            "type": task_type,
            "future": future,
            "start_time": time.time(),
            "message_id": message.id,
        }

        self.status = AgentStatus.BUSY

    def _execute_task(
        self, task_id: str, task_type: str, payload: Dict[str, Any]
    ) -> TaskResult:
        """Execute a specific task"""
        start_time = time.time()

        try:
            result = self.execute_capability(task_type, payload)
            execution_time = time.time() - start_time

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=True,
                result=result,
                execution_time=execution_time,
            )

            self._send_task_response(task_id, task_result)
            return task_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=False,
                error=error_msg,
                execution_time=execution_time,
            )

            self._send_task_response(task_id, task_result)
            logging.error(f"Task {task_id} failed: {error_msg}")
            return task_result

        finally:
            # Clean up completed task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            # Update status
            if not self.active_tasks:
                self.status = AgentStatus.IDLE

    def _send_task_response(self, task_id: str, result: TaskResult):
        """Send task completion response"""
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_RESPONSE,
            sender=self.agent_id,
            recipient="orchestrator",  # Default recipient
            payload={"task_id": task_id, "result": asdict(result)},
            timestamp=time.time(),
        )
        self.message_bus.publish(message)

    def _send_error_response(self, original_message: Message, error: str):
        """Send error response for failed task requests"""
        response = Message(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR_REPORT,
            sender=self.agent_id,
            recipient=original_message.sender,
            payload={"error": error, "original_message_id": original_message.id},
            timestamp=time.time(),
            correlation_id=original_message.id,
        )
        self.message_bus.publish(response)

    def _handle_capability_query(self, message: Message):
        """Respond to capability queries"""
        if message.recipient != self.agent_id and message.recipient != "all":
            return

        response = Message(
            id=str(uuid.uuid4()),
            type=MessageType.CAPABILITY_RESPONSE,
            sender=self.agent_id,
            recipient=message.sender,
            payload={
                "agent_id": self.agent_id,
                "capabilities": [asdict(cap) for cap in self.capabilities.values()],
                "status": self.status.value,
                "active_tasks": len(self.active_tasks),
            },
            timestamp=time.time(),
            correlation_id=message.id,
        )
        self.message_bus.publish(response)

    def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages"""
        self.last_heartbeat = time.time()

    def send_heartbeat(self):
        """Send heartbeat to orchestrator"""
        heartbeat = Message(
            id=str(uuid.uuid4()),
            type=MessageType.HEARTBEAT,
            sender=self.agent_id,
            recipient="orchestrator",
            payload={
                "status": self.status.value,
                "active_tasks": len(self.active_tasks),
                "uptime": time.time() - self.last_heartbeat,
            },
            timestamp=time.time(),
        )
        self.message_bus.publish(heartbeat)

    @abstractmethod
    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute a specific capability - must be implemented by subclasses"""
        pass

    def shutdown(self):
        """Gracefully shutdown the agent"""
        self.status = AgentStatus.STOPPING
        self.executor.shutdown(wait=True)
        self.status = AgentStatus.OFFLINE
        logging.info(f"Agent {self.agent_id} shutdown complete")


class EnhancedOrchestrator:
    """Advanced orchestrator with adaptive planning and load balancing"""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Subscribe to agent messages
        self.message_bus.subscribe(
            MessageType.TASK_RESPONSE, self._handle_task_response
        )
        self.message_bus.subscribe(MessageType.ERROR_REPORT, self._handle_error_report)
        self.message_bus.subscribe(
            MessageType.CAPABILITY_RESPONSE, self._handle_capability_response
        )
        self.message_bus.subscribe(MessageType.HEARTBEAT, self._handle_heartbeat)

        logging.info("Enhanced Orchestrator initialized")

    def discover_agents(self) -> Dict[str, List[AgentCapability]]:
        """Discover available agents and their capabilities"""
        query = Message(
            id=str(uuid.uuid4()),
            type=MessageType.CAPABILITY_QUERY,
            sender="orchestrator",
            recipient="all",
            payload={},
            timestamp=time.time(),
        )

        self.message_bus.publish(query)

        # Wait for responses (in a real system, this would be more sophisticated)
        time.sleep(2)

        return {
            agent_id: info.get("capabilities", [])
            for agent_id, info in self.agents.items()
        }

    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 1,
        timeout: Optional[float] = None,
    ) -> str:
        """Submit a task for execution"""
        task_id = str(uuid.uuid4())

        # Find best agent for this task
        best_agent = self._select_best_agent(task_type)

        if not best_agent:
            raise ValueError(f"No agent available for task type: {task_type}")

        task_request = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            recipient=best_agent,
            payload={"task_id": task_id, "task_type": task_type, **payload},
            timestamp=time.time(),
            priority=priority,
            timeout=timeout,
        )

        self.message_bus.publish(task_request)
        logging.info(f"Task {task_id} submitted to agent {best_agent}")

        return task_id

    def _select_best_agent(self, task_type: str) -> Optional[str]:
        """Select the best agent for a task based on load and capabilities"""
        candidates = []

        for agent_id, info in self.agents.items():
            capabilities = info.get("capabilities", [])

            # Check if agent has required capability
            has_capability = any(cap.get("name") == task_type for cap in capabilities)
            if not has_capability:
                continue

            # Calculate load score (lower is better)
            active_tasks = info.get("active_tasks", 0)
            max_concurrent = next(
                (
                    cap.get("max_concurrent", 1)
                    for cap in capabilities
                    if cap.get("name") == task_type
                ),
                1,
            )

            if active_tasks >= max_concurrent:
                continue  # Agent at capacity

            load_score = active_tasks / max_concurrent
            candidates.append((agent_id, load_score))

        if not candidates:
            return None

        # Return agent with lowest load
        return min(candidates, key=lambda x: x[1])[0]

    def _handle_task_response(self, message: Message):
        """Handle task completion responses"""
        payload = message.payload
        task_id = payload.get("task_id")
        result_data = payload.get("result")

        if result_data:
            result = TaskResult(**result_data)
            self.completed_tasks[task_id] = result

            logging.info(
                f"Task {task_id} completed by {message.sender}: "
                f"{'SUCCESS' if result.success else 'FAILED'}"
            )

    def _handle_error_report(self, message: Message):
        """Handle error reports from agents"""
        logging.error(
            f"Error report from {message.sender}: {message.payload.get('error')}"
        )

    def _handle_capability_response(self, message: Message):
        """Handle capability responses from agents"""
        payload = message.payload
        agent_id = payload.get("agent_id")

        self.agents[agent_id] = {
            "capabilities": payload.get("capabilities", []),
            "status": payload.get("status"),
            "active_tasks": payload.get("active_tasks", 0),
            "last_seen": time.time(),
        }

    def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages from agents"""
        agent_id = message.sender
        if agent_id in self.agents:
            self.agents[agent_id]["last_seen"] = time.time()
            self.agents[agent_id]["status"] = message.payload.get("status")
            self.agents[agent_id]["active_tasks"] = message.payload.get(
                "active_tasks", 0
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "agents": len(self.agents),
            "total_tasks_completed": len(self.completed_tasks),
            "message_bus_stats": self.message_bus.get_stats(),
            "agent_details": self.agents,
        }


# Example agent implementations
class ResearchAgent(BaseAgent):
    """Example research agent implementation"""

    def __init__(self, message_bus: MessageBus):
        capabilities = [
            AgentCapability(
                name="web_research",
                description="Conduct web research on given topics",
                input_schema={"query": "string", "depth": "integer"},
                output_schema={"results": "array", "summary": "string"},
                estimated_duration=30.0,
                max_concurrent=2,
            ),
            AgentCapability(
                name="data_analysis",
                description="Analyze structured data and generate insights",
                input_schema={"data": "object", "analysis_type": "string"},
                output_schema={"insights": "array", "recommendations": "array"},
                estimated_duration=15.0,
                max_concurrent=3,
            ),
        ]
        super().__init__("research_agent", capabilities, message_bus)

    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute research capabilities"""
        if capability_name == "web_research":
            query = payload.get("query")
            depth = payload.get("depth", 1)

            # Simulate research
            time.sleep(2)  # Simulate work
            return {
                "results": [f"Result {i} for query: {query}" for i in range(depth)],
                "summary": f"Research summary for: {query}",
            }

        elif capability_name == "data_analysis":
            data = payload.get("data")
            analysis_type = payload.get("analysis_type", "basic")

            # Simulate analysis
            time.sleep(1)
            return {
                "insights": [f"Insight about {analysis_type} analysis"],
                "recommendations": [f"Recommendation based on {analysis_type}"],
            }

        else:
            raise ValueError(f"Unknown capability: {capability_name}")


# Configuration and factory functions
def create_agent_system() -> tuple[MessageBus, EnhancedOrchestrator, List[BaseAgent]]:
    """Factory function to create a complete agent system"""

    # Create message bus
    message_bus = MessageBus()

    # Create orchestrator
    orchestrator = EnhancedOrchestrator(message_bus)

    # Create agents
    agents = [ResearchAgent(message_bus)]

    return message_bus, orchestrator, agents


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    message_bus, orchestrator, agents = create_agent_system()

    # Discover agents
    print("Discovering agents...")
    capabilities = orchestrator.discover_agents()
    print(f"Found {len(capabilities)} agents")

    # Submit a task
    print("Submitting research task...")
    task_id = orchestrator.submit_task(
        "web_research", {"query": "AI agent frameworks", "depth": 3}
    )

    # Wait for completion
    time.sleep(5)

    # Check results
    if task_id in orchestrator.completed_tasks:
        result = orchestrator.completed_tasks[task_id]
        print(f"Task completed: {result.success}")
        if result.success:
            print(f"Result: {result.result}")

    # Show system status
    status = orchestrator.get_system_status()
    print(f"System status: {status}")

    # Cleanup
    for agent in agents:
        agent.shutdown()
