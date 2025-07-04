"""
Advanced Orchestrator with Adaptive Planning Capabilities
Implements intelligent task planning, execution, and adaptation with real-time optimization.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading
from collections import defaultdict, deque
import heapq


class PlanStatus(Enum):
    """Plan execution status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ADAPTING = "adapting"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class DependencyType(Enum):
    """Types of task dependencies"""

    SEQUENTIAL = "sequential"  # Must complete before next starts
    PARALLEL = "parallel"  # Can run in parallel
    CONDITIONAL = "conditional"  # Depends on result of previous task
    RESOURCE = "resource"  # Shares limited resources


@dataclass
class Task:
    """Individual task definition with metadata"""

    id: str
    name: str
    agent_type: str
    capability: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 30.0
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    required_resources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    status: str = "pending"
    assigned_agent: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ExecutionPlan:
    """Complete execution plan with task dependencies"""

    id: str
    name: str
    description: str
    tasks: List[Task]
    created_time: float
    deadline: Optional[float] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: PlanStatus = PlanStatus.PENDING

    # Execution tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    adaptation_count: int = 0

    # Performance metrics
    original_estimated_duration: float = 0.0
    actual_duration: Optional[float] = None
    efficiency_score: Optional[float] = None


@dataclass
class ResourceConstraint:
    """Resource availability and constraints"""

    name: str
    capacity: int
    current_usage: int = 0
    reserved: Dict[str, int] = field(default_factory=dict)

    def is_available(self, required: int) -> bool:
        return self.current_usage + required <= self.capacity

    def reserve(self, task_id: str, amount: int) -> bool:
        if self.is_available(amount):
            self.reserved[task_id] = amount
            self.current_usage += amount
            return True
        return False

    def release(self, task_id: str):
        if task_id in self.reserved:
            amount = self.reserved.pop(task_id)
            self.current_usage -= amount


@dataclass
class AgentCapacity:
    """Agent capacity and current load"""

    agent_id: str
    max_concurrent: int
    current_tasks: int = 0
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    availability_score: float = 1.0

    def is_available(self) -> bool:
        return self.current_tasks < self.max_concurrent

    def can_handle(self, capability: str) -> bool:
        return capability in self.capabilities


class PlanOptimizer:
    """Optimize execution plans for better performance"""

    def __init__(self):
        self.optimization_strategies = [
            self._optimize_parallel_execution,
            self._optimize_resource_allocation,
            self._optimize_agent_assignment,
            self._optimize_task_ordering,
        ]

    def optimize_plan(
        self,
        plan: ExecutionPlan,
        agent_capacities: Dict[str, AgentCapacity],
        resource_constraints: Dict[str, ResourceConstraint],
    ) -> ExecutionPlan:
        """Optimize execution plan using multiple strategies"""

        optimized_plan = plan

        for strategy in self.optimization_strategies:
            try:
                optimized_plan = strategy(
                    optimized_plan, agent_capacities, resource_constraints
                )
            except Exception as e:
                logging.warning(f"Optimization strategy failed: {e}")

        return optimized_plan

    def _optimize_parallel_execution(
        self,
        plan: ExecutionPlan,
        agent_capacities: Dict[str, AgentCapacity],
        resource_constraints: Dict[str, ResourceConstraint],
    ) -> ExecutionPlan:
        """Identify tasks that can run in parallel"""

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(plan.tasks)

        # Find independent task groups
        parallel_groups = self._find_parallel_groups(dependency_graph)

        # Update dependency types for parallel tasks
        for group in parallel_groups:
            for task_id in group:
                task = next(t for t in plan.tasks if t.id == task_id)
                if len(group) > 1:
                    task.dependency_type = DependencyType.PARALLEL

        return plan

    def _optimize_resource_allocation(
        self,
        plan: ExecutionPlan,
        agent_capacities: Dict[str, AgentCapacity],
        resource_constraints: Dict[str, ResourceConstraint],
    ) -> ExecutionPlan:
        """Optimize resource allocation across tasks"""

        # Sort tasks by priority and resource requirements
        sorted_tasks = sorted(
            plan.tasks,
            key=lambda t: (t.priority.value, len(t.required_resources)),
            reverse=True,
        )

        # Allocate resources optimally
        for task in sorted_tasks:
            best_allocation = self._find_best_resource_allocation(
                task, resource_constraints
            )
            if best_allocation:
                task.required_resources = best_allocation

        return plan

    def _optimize_agent_assignment(
        self,
        plan: ExecutionPlan,
        agent_capacities: Dict[str, AgentCapacity],
        resource_constraints: Dict[str, ResourceConstraint],
    ) -> ExecutionPlan:
        """Optimize agent assignment based on capacity and performance"""

        for task in plan.tasks:
            best_agent = self._select_optimal_agent(task, agent_capacities)
            if best_agent:
                task.assigned_agent = best_agent

        return plan

    def _optimize_task_ordering(
        self,
        plan: ExecutionPlan,
        agent_capacities: Dict[str, AgentCapacity],
        resource_constraints: Dict[str, ResourceConstraint],
    ) -> ExecutionPlan:
        """Optimize task execution order for minimal total time"""

        # Use topological sort with priority weighting
        ordered_tasks = self._topological_sort_with_priority(plan.tasks)
        plan.tasks = ordered_tasks

        return plan

    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """Build task dependency graph"""
        graph = defaultdict(set)
        task_ids = {task.id for task in tasks}

        for task in tasks:
            # Ensure task is in graph even if it has no dependencies
            if task.id not in graph:
                graph[task.id] = set()

            for dep in task.dependencies:
                # Only add dependency if the dependent task exists
                if dep in task_ids:
                    graph[dep].add(task.id)
                else:
                    logging.warning(
                        f"Task {task.id} depends on non-existent task {dep}"
                    )

        return dict(graph)

    def _find_parallel_groups(
        self, dependency_graph: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """Find groups of tasks that can run in parallel"""

        # Find all tasks that are dependencies of others
        dependent_tasks = set()
        for dependents in dependency_graph.values():
            dependent_tasks.update(dependents)

        # Find tasks with no outgoing dependencies (potential parallel tasks)
        all_tasks = set(dependency_graph.keys())
        independent_tasks = all_tasks - dependent_tasks

        # Group independent tasks as a parallel group
        parallel_groups = [independent_tasks] if independent_tasks else []

        return parallel_groups

    def _find_best_resource_allocation(
        self, task: Task, resource_constraints: Dict[str, ResourceConstraint]
    ) -> Optional[List[str]]:
        """Find optimal resource allocation for task"""
        # Simplified resource allocation logic
        available_resources = []

        for resource_name in task.required_resources:
            if resource_name in resource_constraints:
                constraint = resource_constraints[resource_name]
                if constraint.is_available(1):  # Assuming 1 unit needed
                    available_resources.append(resource_name)

        return available_resources if available_resources else None

    def _select_optimal_agent(
        self, task: Task, agent_capacities: Dict[str, AgentCapacity]
    ) -> Optional[str]:
        """Select best agent for task based on capacity and performance"""
        best_agent = None
        best_score = 0.0

        for agent_id, capacity in agent_capacities.items():
            if capacity.can_handle(task.capability) and capacity.is_available():
                # Score based on performance and availability
                score = capacity.performance_score * capacity.availability_score
                score /= max(1, capacity.current_tasks + 1)  # Penalize busy agents

                if score > best_score:
                    best_score = score
                    best_agent = agent_id

        return best_agent

    def _topological_sort_with_priority(self, tasks: List[Task]) -> List[Task]:
        """Topological sort considering task priorities"""
        # Build adjacency list and in-degree count
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)
        task_map = {task.id: task for task in tasks}

        for task in tasks:
            for dep in task.dependencies:
                adj_list[dep].append(task.id)
                in_degree[task.id] += 1

        # Priority queue for tasks with no dependencies
        ready_queue = []
        for task in tasks:
            if in_degree[task.id] == 0:
                heapq.heappush(ready_queue, (-task.priority.value, task.id))

        sorted_tasks = []

        while ready_queue:
            _, task_id = heapq.heappop(ready_queue)
            sorted_tasks.append(task_map[task_id])

            # Update dependencies
            for dependent_id in adj_list[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    dependent_task = task_map[dependent_id]
                    heapq.heappush(
                        ready_queue, (-dependent_task.priority.value, dependent_id)
                    )

        return sorted_tasks


class AdaptivePlanner:
    """Dynamic plan adaptation based on execution feedback"""

    def __init__(self):
        self.adaptation_strategies = [
            self._handle_task_failure,
            self._handle_resource_shortage,
            self._handle_performance_degradation,
            self._handle_deadline_pressure,
        ]

    def adapt_plan(
        self, plan: ExecutionPlan, execution_context: Dict[str, Any]
    ) -> Optional[ExecutionPlan]:
        """Adapt plan based on execution context"""

        adaptation_needed = self._assess_adaptation_need(plan, execution_context)

        if not adaptation_needed:
            return None

        adapted_plan = self._create_adapted_plan(plan, execution_context)
        adapted_plan.adaptation_count += 1

        logging.info(
            f"Plan {plan.id} adapted (adaptation #{adapted_plan.adaptation_count})"
        )

        return adapted_plan

    def _assess_adaptation_need(
        self, plan: ExecutionPlan, execution_context: Dict[str, Any]
    ) -> bool:
        """Determine if plan adaptation is needed"""

        # Check for failed tasks
        if execution_context.get("failed_tasks"):
            return True

        # Check for resource constraints
        if execution_context.get("resource_shortages"):
            return True

        # Check for performance degradation
        performance_ratio = execution_context.get("performance_ratio", 1.0)
        if performance_ratio < 0.7:  # 30% degradation
            return True

        # Check deadline pressure
        if execution_context.get("deadline_pressure", 0.0) > 0.8:
            return True

        return False

    def _create_adapted_plan(
        self, original_plan: ExecutionPlan, execution_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Create adapted version of the plan"""

        adapted_plan = ExecutionPlan(
            id=f"{original_plan.id}_adapted_{original_plan.adaptation_count + 1}",
            name=f"{original_plan.name} (Adapted)",
            description=original_plan.description,
            tasks=original_plan.tasks.copy(),
            created_time=time.time(),
            deadline=original_plan.deadline,
            priority=original_plan.priority,
            status=PlanStatus.ADAPTING,
        )

        # Apply adaptation strategies
        for strategy in self.adaptation_strategies:
            try:
                strategy(adapted_plan, execution_context)
            except Exception as e:
                logging.warning(f"Adaptation strategy failed: {e}")

        return adapted_plan

    def _handle_task_failure(self, plan: ExecutionPlan, context: Dict[str, Any]):
        """Handle failed tasks by adding retries or alternatives"""
        failed_tasks = context.get("failed_tasks", [])

        for task_id in failed_tasks:
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if task and task.retry_count < task.max_retries:
                # Reset task for retry
                task.status = "pending"
                task.assigned_agent = None
                task.start_time = None
                task.end_time = None
                task.error = None
                task.retry_count += 1

                # Increase timeout for retry
                if task.timeout:
                    task.timeout *= 1.5

    def _handle_resource_shortage(self, plan: ExecutionPlan, context: Dict[str, Any]):
        """Handle resource shortages by rescheduling or finding alternatives"""
        resource_shortages = context.get("resource_shortages", [])

        for resource_name in resource_shortages:
            # Find tasks that require this resource
            affected_tasks = [
                t for t in plan.tasks if resource_name in t.required_resources
            ]

            # Reschedule to spread resource usage over time
            for i, task in enumerate(affected_tasks):
                task.estimated_duration += i * 5  # Add delay

    def _handle_performance_degradation(
        self, plan: ExecutionPlan, context: Dict[str, Any]
    ):
        """Handle performance issues by redistributing load"""
        performance_ratio = context.get("performance_ratio", 1.0)

        if performance_ratio < 0.7:
            # Reduce parallel execution
            for task in plan.tasks:
                if task.dependency_type == DependencyType.PARALLEL:
                    task.dependency_type = DependencyType.SEQUENTIAL

    def _handle_deadline_pressure(self, plan: ExecutionPlan, context: Dict[str, Any]):
        """Handle deadline pressure by prioritizing critical tasks"""
        deadline_pressure = context.get("deadline_pressure", 0.0)

        if deadline_pressure > 0.8:
            # Increase priority of pending tasks
            for task in plan.tasks:
                if task.status == "pending" and task.priority != TaskPriority.CRITICAL:
                    task.priority = TaskPriority.HIGH


class AdvancedOrchestrator:
    """Advanced orchestrator with adaptive planning and optimization"""

    def __init__(self, message_bus, tool_registry):
        self.message_bus = message_bus
        self.tool_registry = tool_registry

        # Core components
        self.plan_optimizer = PlanOptimizer()
        self.adaptive_planner = AdaptivePlanner()

        # State management
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.agent_capacities: Dict[str, AgentCapacity] = {}
        self.resource_constraints: Dict[str, ResourceConstraint] = {}

        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.execution_monitor = threading.Thread(
            target=self._monitor_execution, daemon=True
        )
        self.execution_monitor.start()

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = []

        logging.info("Advanced Orchestrator initialized")

    def create_plan(
        self,
        name: str,
        description: str,
        tasks: List[Dict[str, Any]],
        deadline: Optional[float] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """Create a new execution plan"""

        plan_id = str(uuid.uuid4())

        # Convert task dictionaries to Task objects
        task_objects = []
        for i, task_data in enumerate(tasks):
            task = Task(
                id=task_data.get("id", f"{plan_id}_task_{i}"),
                name=task_data["name"],
                agent_type=task_data["agent_type"],
                capability=task_data["capability"],
                parameters=task_data.get("parameters", {}),
                priority=TaskPriority(
                    task_data.get("priority", TaskPriority.NORMAL.value)
                ),
                estimated_duration=task_data.get("estimated_duration", 30.0),
                max_retries=task_data.get("max_retries", 3),
                timeout=task_data.get("timeout"),
                dependencies=task_data.get("dependencies", []),
                dependency_type=DependencyType(
                    task_data.get("dependency_type", "sequential")
                ),
                required_resources=task_data.get("required_resources", []),
                tags=task_data.get("tags", []),
                conditions=task_data.get("conditions", {}),
            )
            task_objects.append(task)

        # Create execution plan
        plan = ExecutionPlan(
            id=plan_id,
            name=name,
            description=description,
            tasks=task_objects,
            created_time=time.time(),
            deadline=deadline,
            priority=priority,
            original_estimated_duration=sum(t.estimated_duration for t in task_objects),
        )

        # Optimize plan
        optimized_plan = self.plan_optimizer.optimize_plan(
            plan, self.agent_capacities, self.resource_constraints
        )

        self.active_plans[plan_id] = optimized_plan

        logging.info(f"Created optimized plan {plan_id} with {len(task_objects)} tasks")
        return plan_id

    def execute_plan(self, plan_id: str) -> bool:
        """Start executing a plan"""

        if plan_id not in self.active_plans:
            logging.error(f"Plan {plan_id} not found")
            return False

        plan = self.active_plans[plan_id]
        plan.status = PlanStatus.IN_PROGRESS
        plan.start_time = time.time()

        # Submit plan for execution
        future = self.executor.submit(self._execute_plan_tasks, plan)

        logging.info(f"Started executing plan {plan_id}")
        return True

    def _execute_plan_tasks(self, plan: ExecutionPlan):
        """Execute all tasks in a plan"""

        try:
            # Build execution schedule
            schedule = self._build_execution_schedule(plan)

            # Execute tasks according to schedule
            for batch in schedule:
                batch_futures = []

                for task in batch:
                    if self._can_execute_task(task, plan):
                        future = self.executor.submit(self._execute_task, task, plan)
                        batch_futures.append((task, future))

                # Wait for batch completion
                for task, future in batch_futures:
                    try:
                        result = future.result(timeout=task.timeout)
                        self._handle_task_completion(task, result, plan)
                    except Exception as e:
                        self._handle_task_failure(task, e, plan)

                # Check if adaptation is needed
                self._check_adaptation_need(plan)

            # Plan completed
            plan.status = PlanStatus.COMPLETED
            plan.end_time = time.time()
            plan.actual_duration = plan.end_time - plan.start_time
            plan.efficiency_score = (
                plan.original_estimated_duration / plan.actual_duration
            )

            logging.info(f"Plan {plan.id} completed successfully")

        except Exception as e:
            plan.status = PlanStatus.FAILED
            plan.end_time = time.time()
            logging.error(f"Plan {plan.id} failed: {e}")

    def _build_execution_schedule(self, plan: ExecutionPlan) -> List[List[Task]]:
        """Build execution schedule respecting dependencies"""

        schedule = []
        completed_tasks = set()
        remaining_tasks = plan.tasks.copy()

        while remaining_tasks:
            ready_tasks = []

            # Find tasks with met dependencies
            for task in remaining_tasks:
                if all(dep in completed_tasks for dep in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                logging.error(f"Dependency deadlock in plan {plan.id}")
                break

            # Separate parallelizable and sequential tasks
            parallel_batch = [
                t for t in ready_tasks if t.dependency_type == DependencyType.PARALLEL
            ]
            sequential_tasks = [
                t for t in ready_tasks if t.dependency_type != DependencyType.PARALLEL
            ]

            # Add parallel tasks as a single batch
            if parallel_batch:
                schedule.append(parallel_batch)
                for task in parallel_batch:
                    remaining_tasks.remove(task)
                    completed_tasks.add(task.id)

            # Add sequential tasks in individual batches
            for task in sequential_tasks:
                schedule.append([task])
                remaining_tasks.remove(task)
                completed_tasks.add(task.id)

        return schedule

    def _can_execute_task(self, task: Task, plan: ExecutionPlan) -> bool:
        """Check if task can be executed now"""

        # Check agent availability
        if task.assigned_agent:
            agent_capacity = self.agent_capacities.get(task.assigned_agent)
            if not agent_capacity or not agent_capacity.is_available():
                return False

        # Check resource availability
        for resource_name in task.required_resources:
            if resource_name in self.resource_constraints:
                constraint = self.resource_constraints[resource_name]
                if not constraint.is_available(1):
                    return False

        return True

    def _execute_task(self, task: Task, plan: ExecutionPlan) -> Any:
        """Execute a single task"""

        task.status = "running"
        task.start_time = time.time()

        try:
            # Reserve resources
            for resource_name in task.required_resources:
                if resource_name in self.resource_constraints:
                    self.resource_constraints[resource_name].reserve(task.id, 1)

            # Execute task using appropriate tool/agent
            if task.capability in self.tool_registry.tools:
                result = self.tool_registry.execute_tool(
                    task.capability, **task.parameters
                )
            else:
                # Send to agent via message bus
                result = self._send_task_to_agent(task)

            task.result = result
            task.status = "completed"
            task.end_time = time.time()

            return result

        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.end_time = time.time()
            raise e

        finally:
            # Release resources
            for resource_name in task.required_resources:
                if resource_name in self.resource_constraints:
                    self.resource_constraints[resource_name].release(task.id)

    def _send_task_to_agent(self, task: Task) -> Any:
        """Send task to appropriate agent via message bus"""
        # Implementation would depend on specific message bus setup
        # This is a placeholder for the actual agent communication
        time.sleep(task.estimated_duration)  # Simulate execution
        return {"status": "completed", "result": f"Task {task.name} completed"}

    def _handle_task_completion(self, task: Task, result: Any, plan: ExecutionPlan):
        """Handle successful task completion"""
        plan.completed_tasks.add(task.id)

        # Update agent capacity
        if task.assigned_agent and task.assigned_agent in self.agent_capacities:
            self.agent_capacities[task.assigned_agent].current_tasks -= 1

        # Update the active plan in the registry to reflect completion
        if plan.id in self.active_plans:
            self.active_plans[plan.id].completed_tasks.add(task.id)

        logging.info(f"Task {task.id} completed successfully")

    def _handle_task_failure(self, task: Task, error: Exception, plan: ExecutionPlan):
        """Handle task failure"""
        plan.failed_tasks.add(task.id)

        logging.error(f"Task {task.id} failed: {error}")

        # Update agent capacity
        if task.assigned_agent and task.assigned_agent in self.agent_capacities:
            self.agent_capacities[task.assigned_agent].current_tasks -= 1

    def _check_adaptation_need(self, plan: ExecutionPlan):
        """Check if plan needs adaptation"""

        execution_context = {
            "failed_tasks": list(plan.failed_tasks),
            "completed_tasks": list(plan.completed_tasks),
            "performance_ratio": self._calculate_performance_ratio(plan),
            "deadline_pressure": self._calculate_deadline_pressure(plan),
            "resource_shortages": self._identify_resource_shortages(),
        }

        adapted_plan = self.adaptive_planner.adapt_plan(plan, execution_context)

        if adapted_plan:
            # Preserve execution state in adapted plan
            adapted_plan.completed_tasks = plan.completed_tasks.copy()
            adapted_plan.failed_tasks = plan.failed_tasks.copy()
            adapted_plan.start_time = plan.start_time
            adapted_plan.status = plan.status

            # Replace current plan with adapted version
            self.active_plans[plan.id] = adapted_plan
            self.adaptation_history.append(
                {
                    "original_plan_id": plan.id,
                    "adapted_plan_id": adapted_plan.id,
                    "timestamp": time.time(),
                    "context": execution_context,
                }
            )

    def _calculate_performance_ratio(self, plan: ExecutionPlan) -> float:
        """Calculate current performance vs expected"""
        if not plan.start_time:
            return 1.0

        elapsed = time.time() - plan.start_time
        completed_ratio = len(plan.completed_tasks) / len(plan.tasks)
        expected_progress = elapsed / plan.original_estimated_duration

        if expected_progress == 0:
            return 1.0

        return completed_ratio / expected_progress

    def _calculate_deadline_pressure(self, plan: ExecutionPlan) -> float:
        """Calculate deadline pressure (0.0 to 1.0)"""
        if not plan.deadline or not plan.start_time:
            return 0.0

        elapsed = time.time() - plan.start_time
        total_time = plan.deadline - plan.start_time

        if total_time <= 0:
            return 1.0

        return elapsed / total_time

    def _identify_resource_shortages(self) -> List[str]:
        """Identify resources that are running low"""
        shortages = []

        for resource_name, constraint in self.resource_constraints.items():
            utilization = constraint.current_usage / constraint.capacity
            if utilization > 0.9:  # 90% utilization threshold
                shortages.append(resource_name)

        return shortages

    def _monitor_execution(self):
        """Background monitoring of plan execution"""
        while True:
            try:
                for plan_id, plan in self.active_plans.items():
                    if plan.status == PlanStatus.IN_PROGRESS:
                        self._update_plan_metrics(plan)

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logging.error(f"Execution monitoring error: {e}")

    def _update_plan_metrics(self, plan: ExecutionPlan):
        """Update plan performance metrics"""
        # This would update various metrics and trigger adaptations if needed
        pass

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a plan"""
        if plan_id not in self.active_plans:
            return None

        plan = self.active_plans[plan_id]

        return {
            "plan_id": plan.id,
            "name": plan.name,
            "status": plan.status.value,
            "progress": len(plan.completed_tasks) / len(plan.tasks),
            "completed_tasks": len(plan.completed_tasks),
            "failed_tasks": len(plan.failed_tasks),
            "total_tasks": len(plan.tasks),
            "start_time": plan.start_time,
            "estimated_completion": self._estimate_completion_time(plan),
            "adaptation_count": plan.adaptation_count,
        }

    def _estimate_completion_time(self, plan: ExecutionPlan) -> Optional[float]:
        """Estimate when plan will complete"""
        if not plan.start_time or plan.status != PlanStatus.IN_PROGRESS:
            return None

        completed_ratio = len(plan.completed_tasks) / len(plan.tasks)
        if completed_ratio == 0:
            return None

        elapsed = time.time() - plan.start_time
        estimated_total = elapsed / completed_ratio

        return plan.start_time + estimated_total

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "active_plans": len(
                [
                    p
                    for p in self.active_plans.values()
                    if p.status == PlanStatus.IN_PROGRESS
                ]
            ),
            "total_plans": len(self.active_plans),
            "agent_utilization": {
                agent_id: capacity.current_tasks / capacity.max_concurrent
                for agent_id, capacity in self.agent_capacities.items()
            },
            "resource_utilization": {
                name: constraint.current_usage / constraint.capacity
                for name, constraint in self.resource_constraints.items()
            },
            "adaptation_count": len(self.adaptation_history),
            "performance_history": list(self.performance_history),
        }


# Example usage functions
def create_sample_plan() -> Dict[str, Any]:
    """Create a sample execution plan for testing"""

    # Generate unique task IDs
    task_1_id = "task_research_market_trends"
    task_2_id = "task_analyze_competitor_data"
    task_3_id = "task_generate_report"

    tasks = [
        {
            "id": task_1_id,
            "name": "Research Market Trends",
            "agent_type": "research_agent",
            "capability": "web_research",
            "parameters": {"query": "AI market trends 2024", "depth": 3},
            "priority": TaskPriority.HIGH.value,
            "estimated_duration": 5.0,  # Reduced for faster testing
            "dependencies": [],
            "required_resources": ["internet_access"],
        },
        {
            "id": task_2_id,
            "name": "Analyze Competitor Data",
            "agent_type": "research_agent",
            "capability": "data_analysis",
            "parameters": {"analysis_type": "competitive"},
            "priority": TaskPriority.NORMAL.value,
            "estimated_duration": 3.0,  # Reduced for faster testing
            "dependencies": [task_1_id],  # Use task ID instead of name
            "required_resources": ["cpu_intensive"],
        },
        {
            "id": task_3_id,
            "name": "Generate Report",
            "agent_type": "report_writer",
            "capability": "create_report",
            "parameters": {"format": "executive_summary"},
            "priority": TaskPriority.HIGH.value,
            "estimated_duration": 2.0,  # Reduced for faster testing
            "dependencies": [task_2_id],  # Use task ID instead of name
            "required_resources": ["document_generator"],
        },
    ]

    return {
        "name": "Market Analysis Project",
        "description": "Comprehensive market analysis with competitive intelligence",
        "tasks": tasks,
        "deadline": time.time() + 3600,  # 1 hour from now
        "priority": TaskPriority.HIGH,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # This would be integrated with the actual message bus and tool registry
    # For demonstration purposes, we'll create mock objects

    from tool_integration_system import create_tool_system
    from agent_framework import create_agent_system

    # Create supporting systems
    tool_registry = create_tool_system()
    message_bus, _, _ = create_agent_system()

    # Create orchestrator
    orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

    # Add some agent capacities and resource constraints
    orchestrator.agent_capacities["research_agent"] = AgentCapacity(
        agent_id="research_agent",
        max_concurrent=3,
        capabilities=["web_research", "data_analysis"],
        performance_score=0.9,
    )

    orchestrator.resource_constraints["internet_access"] = ResourceConstraint(
        name="internet_access", capacity=10
    )

    # Create and execute sample plan
    plan_data = create_sample_plan()
    plan_id = orchestrator.create_plan(**plan_data)

    print(f"Created plan: {plan_id}")

    # Start execution
    orchestrator.execute_plan(plan_id)

    # Monitor progress
    time.sleep(2)
    status = orchestrator.get_plan_status(plan_id)
    print(f"Plan status: {status}")

    # Get system metrics
    metrics = orchestrator.get_system_metrics()
    print(f"System metrics: {json.dumps(metrics, indent=2)}")
