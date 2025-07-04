"""
Enhanced Tool Integration and Error Handling System
Provides robust tool management, error recovery, and performance monitoring capabilities.
"""

import asyncio
import json
import logging
import time
import inspect
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Type
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from functools import wraps


class ToolStatus(Enum):
    """Tool operational status"""

    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolDefinition:
    """Comprehensive tool definition with metadata"""

    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: Optional[int] = None
    error_handlers: Dict[str, Callable] = field(default_factory=dict)
    health_check: Optional[Callable] = None
    documentation_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolExecution:
    """Track tool execution details"""

    tool_name: str
    execution_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorContext:
    """Context information for error handling"""

    tool_name: str
    execution_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    retry_count: int
    timestamp: float
    stack_trace: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """Base exception for tool-related errors"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}


class RetryableError(ToolError):
    """Error that can be retried"""

    pass


class NonRetryableError(ToolError):
    """Error that should not be retried"""

    pass


class CircuitBreakerError(ToolError):
    """Error indicating circuit breaker is open"""

    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for tool reliability"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                    self.success_count = 0
                else:
                    raise CircuitBreakerError("Circuit breaker is open")

            try:
                result = func(*args, **kwargs)

                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0

                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

                raise e


class RateLimiter:
    """Token bucket rate limiter for tool calls"""

    def __init__(self, rate: int, burst: int = None):
        self.rate = rate  # tokens per second
        self.burst = burst or rate
        self.tokens = self.burst
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        with self._lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False


class PerformanceMonitor:
    """Monitor tool performance and collect metrics"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.executions: List[ToolExecution] = []
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_duration": 0.0,
            "error_rate": 0.0,
        }
        self._lock = threading.Lock()

    def record_execution(self, execution: ToolExecution):
        """Record a tool execution"""
        with self._lock:
            self.executions.append(execution)

            # Keep only recent executions
            if len(self.executions) > self.window_size:
                self.executions = self.executions[-self.window_size :]

            self._update_metrics()

    def _update_metrics(self):
        """Update performance metrics"""
        if not self.executions:
            return

        total = len(self.executions)
        successful = sum(1 for e in self.executions if e.success)
        failed = total - successful

        durations = [
            e.end_time - e.start_time for e in self.executions if e.end_time is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        self.metrics.update(
            {
                "total_calls": total,
                "successful_calls": successful,
                "failed_calls": failed,
                "average_duration": avg_duration,
                "error_rate": failed / total if total > 0 else 0.0,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.metrics.copy()


class BaseTool(ABC):
    """Base class for all tools with enhanced error handling"""

    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self.status = ToolStatus.AVAILABLE
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = (
            RateLimiter(definition.rate_limit) if definition.rate_limit else None
        )
        self.performance_monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=5)

        logging.info(f"Tool {definition.name} v{definition.version} initialized")

    def execute(self, **kwargs) -> Any:
        """Execute tool with comprehensive error handling"""
        execution_id = f"{self.definition.name}_{int(time.time() * 1000)}"
        execution = ToolExecution(
            tool_name=self.definition.name,
            execution_id=execution_id,
            start_time=time.time(),
        )

        try:
            # Rate limiting
            if self.rate_limiter and not self.rate_limiter.acquire():
                raise ToolError("Rate limit exceeded", ErrorSeverity.LOW)

            # Validate inputs
            self._validate_inputs(kwargs)

            # Execute with circuit breaker and timeout
            future = self.executor.submit(self._execute_with_circuit_breaker, **kwargs)
            result = future.result(timeout=self.definition.timeout)

            # Record successful execution
            execution.end_time = time.time()
            execution.success = True
            execution.result = result

            self.performance_monitor.record_execution(execution)
            return result

        except Exception as e:
            execution.end_time = time.time()
            execution.error = str(e)

            # Handle error with retry logic
            error_context = ErrorContext(
                tool_name=self.definition.name,
                execution_id=execution_id,
                error_type=type(e).__name__,
                error_message=str(e),
                severity=getattr(e, "severity", ErrorSeverity.MEDIUM),
                retry_count=execution.retry_count,
                timestamp=time.time(),
                stack_trace=traceback.format_exc(),
                input_params=kwargs,
            )

            if self._should_retry(e, execution.retry_count):
                return self._retry_execution(execution, error_context, **kwargs)

            self._handle_error(error_context)
            self.performance_monitor.record_execution(execution)
            raise e

    def _execute_with_circuit_breaker(self, **kwargs) -> Any:
        """Execute tool with circuit breaker protection"""
        return self.circuit_breaker.call(self._execute, **kwargs)

    def _validate_inputs(self, inputs: Dict[str, Any]):
        """Validate input parameters against schema"""
        schema = self.definition.input_schema

        # Basic validation - in production, use a proper schema validator
        for field, field_type in schema.items():
            if field in inputs:
                value = inputs[field]
                # Add type checking logic here
                pass

    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if error should be retried"""
        if retry_count >= self.definition.max_retries:
            return False

        if isinstance(error, NonRetryableError):
            return False

        if isinstance(error, (RetryableError, TimeoutError)):
            return True

        # Default retry logic for common errors
        retryable_errors = (ConnectionError, TimeoutError, ToolError)
        return isinstance(error, retryable_errors)

    def _retry_execution(
        self, execution: ToolExecution, error_context: ErrorContext, **kwargs
    ) -> Any:
        """Retry tool execution with exponential backoff"""
        execution.retry_count += 1

        # Exponential backoff
        delay = min(2**execution.retry_count, 30)
        time.sleep(delay)

        logging.warning(
            f"Retrying {self.definition.name} (attempt {execution.retry_count})"
        )
        return self.execute(**kwargs)

    def _handle_error(self, error_context: ErrorContext):
        """Handle errors using registered handlers"""
        error_type = error_context.error_type

        if error_type in self.definition.error_handlers:
            handler = self.definition.error_handlers[error_type]
            try:
                handler(error_context)
            except Exception as e:
                logging.error(f"Error handler failed: {e}")

        # Log error
        logging.error(
            f"Tool {self.definition.name} failed: {error_context.error_message}"
        )

        # Update tool status based on error severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.status = ToolStatus.ERROR

    def health_check(self) -> bool:
        """Check tool health"""
        if self.definition.health_check:
            try:
                return self.definition.health_check()
            except Exception:
                return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get tool status and metrics"""
        return {
            "name": self.definition.name,
            "version": self.definition.version,
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker.state,
            "performance_metrics": self.performance_monitor.get_metrics(),
            "health_check": self.health_check(),
        }

    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """Execute the actual tool logic - must be implemented by subclasses"""
        pass


class ToolRegistry:
    """Central registry for managing tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_dependencies: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        with self._lock:
            name = tool.definition.name
            self.tools[name] = tool
            self.tool_dependencies[name] = tool.definition.dependencies

            logging.info(f"Tool {name} registered successfully")

    def unregister_tool(self, tool_name: str):
        """Unregister a tool"""
        with self._lock:
            if tool_name in self.tools:
                del self.tools[tool_name]
                del self.tool_dependencies[tool_name]
                logging.info(f"Tool {tool_name} unregistered")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolError(f"Tool '{tool_name}' not found", ErrorSeverity.HIGH)

        if tool.status != ToolStatus.AVAILABLE:
            raise ToolError(
                f"Tool '{tool_name}' is not available", ErrorSeverity.MEDIUM
            )

        return tool.execute(**kwargs)

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [
            name
            for name, tool in self.tools.items()
            if tool.status == ToolStatus.AVAILABLE
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        tools_status = {}
        for name, tool in self.tools.items():
            tools_status[name] = tool.get_status()

        return {
            "total_tools": len(self.tools),
            "available_tools": len(self.get_available_tools()),
            "tools": tools_status,
        }

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate tool dependencies"""
        missing_deps = {}

        for tool_name, deps in self.tool_dependencies.items():
            missing = [dep for dep in deps if dep not in self.tools]
            if missing:
                missing_deps[tool_name] = missing

        return missing_deps


# Example tool implementations
class WebSearchTool(BaseTool):
    """Example web search tool"""

    def __init__(self):
        definition = ToolDefinition(
            name="web_search",
            version="1.0.0",
            description="Search the web for information",
            input_schema={"query": "string", "max_results": "integer"},
            output_schema={"results": "array"},
            timeout=15.0,
            max_retries=2,
            rate_limit=10,  # 10 requests per second
        )
        super().__init__(definition)

    def _execute(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Execute web search"""
        if not query:
            raise NonRetryableError("Query cannot be empty", ErrorSeverity.HIGH)

        # Simulate search
        time.sleep(1)  # Simulate API call

        if "error" in query.lower():
            raise RetryableError("Search API error", ErrorSeverity.MEDIUM)

        return {
            "results": [f"Result {i} for query: {query}" for i in range(max_results)],
            "query": query,
            "total_results": max_results,
        }


class DataProcessingTool(BaseTool):
    """Example data processing tool"""

    def __init__(self):
        definition = ToolDefinition(
            name="data_processing",
            version="2.1.0",
            description="Process and analyze data",
            input_schema={"data": "object", "operation": "string"},
            output_schema={"processed_data": "object", "statistics": "object"},
            timeout=30.0,
            max_retries=3,
        )
        super().__init__(definition)

    def _execute(
        self, data: Dict[str, Any], operation: str = "analyze"
    ) -> Dict[str, Any]:
        """Execute data processing"""
        if not data:
            raise NonRetryableError("Data cannot be empty", ErrorSeverity.HIGH)

        # Simulate processing
        time.sleep(2)

        return {
            "processed_data": {"result": f"Processed with {operation}"},
            "statistics": {"rows": len(data), "operation": operation},
        }


# Enhanced error handling decorators
def handle_tool_errors(default_return=None, severity=ErrorSeverity.MEDIUM):
    """Decorator to handle tool errors gracefully"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Tool error in {func.__name__}: {e}")
                if default_return is not None:
                    return default_return
                raise ToolError(str(e), severity)

        return wrapper

    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to tool functions"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                raise ToolError(
                    f"Function {func.__name__} timed out after {timeout_seconds}s"
                )
            finally:
                executor.shutdown(wait=False)

        return wrapper

    return decorator


# Factory function for creating tool system
def create_tool_system() -> ToolRegistry:
    """Create a complete tool integration system"""
    registry = ToolRegistry()

    # Register example tools
    registry.register_tool(WebSearchTool())
    registry.register_tool(DataProcessingTool())

    return registry


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create tool system
    registry = create_tool_system()

    # Test tool execution
    try:
        result = registry.execute_tool("web_search", query="AI tools", max_results=5)
        print(f"Search result: {result}")

        # Test error handling
        error_result = registry.execute_tool("web_search", query="error test")

    except ToolError as e:
        print(f"Tool error: {e}")

    # Show system status
    status = registry.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
