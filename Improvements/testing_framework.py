"""
Comprehensive Testing and Validation Framework
Provides automated testing, validation, and quality assurance for agent systems.
"""

import asyncio
import json
import logging
import time
import inspect
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading
from collections import defaultdict, deque
import tempfile
import shutil
import os


class TestType(Enum):
    """Types of tests in the framework"""

    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    SECURITY = "security"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestSeverity(Enum):
    """Test failure severity"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestAssertion:
    """Individual test assertion"""

    description: str
    actual: Any
    expected: Any
    operator: str  # equals, not_equals, greater_than, less_than, contains, etc.
    passed: bool = False
    error_message: Optional[str] = None


@dataclass
class TestResult:
    """Result of a single test execution"""

    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    assertions: List[TestAssertion] = field(default_factory=list)
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(
        default_factory=dict
    )  # file paths to test artifacts


@dataclass
class TestSuite:
    """Collection of related tests"""

    name: str
    description: str
    tests: List["BaseTest"] = field(default_factory=list)
    setup_hooks: List[Callable] = field(default_factory=list)
    teardown_hooks: List[Callable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)


@dataclass
class TestReport:
    """Comprehensive test execution report"""

    suite_name: str
    execution_id: str
    start_time: float
    end_time: Optional[float] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    test_results: List[TestResult] = field(default_factory=list)
    coverage_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts_directory: Optional[str] = None


class BaseTest(ABC):
    """Base class for all tests"""

    def __init__(
        self,
        test_id: str,
        name: str,
        test_type: TestType,
        description: str = "",
        timeout: float = 30.0,
        requirements: List[str] = None,
        tags: List[str] = None,
    ):
        self.test_id = test_id
        self.name = name
        self.test_type = test_type
        self.description = description
        self.timeout = timeout
        self.requirements = requirements or []
        self.tags = tags or []

        self.setup_hooks: List[Callable] = []
        self.teardown_hooks: List[Callable] = []
        self.assertions: List[TestAssertion] = []

    def add_setup_hook(self, hook: Callable):
        """Add setup hook to run before test"""
        self.setup_hooks.append(hook)

    def add_teardown_hook(self, hook: Callable):
        """Add teardown hook to run after test"""
        self.teardown_hooks.append(hook)

    def assert_equals(self, actual: Any, expected: Any, description: str = ""):
        """Assert that actual equals expected"""
        assertion = TestAssertion(
            description=description or f"Expected {expected}, got {actual}",
            actual=actual,
            expected=expected,
            operator="equals",
            passed=actual == expected,
        )

        if not assertion.passed:
            assertion.error_message = f"Assertion failed: {actual} != {expected}"

        self.assertions.append(assertion)
        return assertion.passed

    def assert_not_equals(self, actual: Any, expected: Any, description: str = ""):
        """Assert that actual does not equal expected"""
        assertion = TestAssertion(
            description=description or f"Expected {actual} != {expected}",
            actual=actual,
            expected=expected,
            operator="not_equals",
            passed=actual != expected,
        )

        if not assertion.passed:
            assertion.error_message = f"Assertion failed: {actual} == {expected}"

        self.assertions.append(assertion)
        return assertion.passed

    def assert_greater_than(
        self, actual: float, expected: float, description: str = ""
    ):
        """Assert that actual is greater than expected"""
        assertion = TestAssertion(
            description=description or f"Expected {actual} > {expected}",
            actual=actual,
            expected=expected,
            operator="greater_than",
            passed=actual > expected,
        )

        if not assertion.passed:
            assertion.error_message = f"Assertion failed: {actual} <= {expected}"

        self.assertions.append(assertion)
        return assertion.passed

    def assert_less_than(self, actual: float, expected: float, description: str = ""):
        """Assert that actual is less than expected"""
        assertion = TestAssertion(
            description=description or f"Expected {actual} < {expected}",
            actual=actual,
            expected=expected,
            operator="less_than",
            passed=actual < expected,
        )

        if not assertion.passed:
            assertion.error_message = f"Assertion failed: {actual} >= {expected}"

        self.assertions.append(assertion)
        return assertion.passed

    def assert_contains(self, container: Any, item: Any, description: str = ""):
        """Assert that container contains item"""
        assertion = TestAssertion(
            description=description or f"Expected {container} to contain {item}",
            actual=container,
            expected=item,
            operator="contains",
            passed=item in container,
        )

        if not assertion.passed:
            assertion.error_message = f"Assertion failed: {item} not in {container}"

        self.assertions.append(assertion)
        return assertion.passed

    def assert_raises(self, exception_type: type, func: Callable, *args, **kwargs):
        """Assert that function raises specific exception"""
        try:
            func(*args, **kwargs)
            # If we get here, no exception was raised
            assertion = TestAssertion(
                description=f"Expected {exception_type.__name__} to be raised",
                actual="No exception",
                expected=exception_type.__name__,
                operator="raises",
                passed=False,
                error_message=f"Expected {exception_type.__name__} but no exception was raised",
            )
        except exception_type:
            # Expected exception was raised
            assertion = TestAssertion(
                description=f"Expected {exception_type.__name__} to be raised",
                actual=exception_type.__name__,
                expected=exception_type.__name__,
                operator="raises",
                passed=True,
            )
        except Exception as e:
            # Wrong exception type was raised
            assertion = TestAssertion(
                description=f"Expected {exception_type.__name__} to be raised",
                actual=type(e).__name__,
                expected=exception_type.__name__,
                operator="raises",
                passed=False,
                error_message=f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}",
            )

        self.assertions.append(assertion)
        return assertion.passed

    @abstractmethod
    def execute(self) -> None:
        """Execute the test - must be implemented by subclasses"""
        pass


class AgentTest(BaseTest):
    """Test for agent functionality"""

    def __init__(self, test_id: str, name: str, agent_system, **kwargs):
        super().__init__(test_id, name, TestType.INTEGRATION, **kwargs)
        self.agent_system = agent_system

    def test_agent_communication(
        self, agent_id: str, message_type: str, payload: Dict[str, Any]
    ):
        """Test agent communication"""
        # Implementation would depend on specific agent system
        pass

    def test_agent_capability(
        self, agent_id: str, capability: str, parameters: Dict[str, Any]
    ):
        """Test specific agent capability"""
        # Implementation would depend on specific agent system
        pass


class ToolTest(BaseTest):
    """Test for tool functionality"""

    def __init__(self, test_id: str, name: str, tool_registry, **kwargs):
        super().__init__(test_id, name, TestType.UNIT, **kwargs)
        self.tool_registry = tool_registry

    def test_tool_execution(
        self, tool_name: str, parameters: Dict[str, Any], expected_result: Any = None
    ):
        """Test tool execution"""
        try:
            result = self.tool_registry.execute_tool(tool_name, **parameters)

            if expected_result is not None:
                self.assert_equals(result, expected_result, f"Tool {tool_name} result")

            return result

        except Exception as e:
            self.assertions.append(
                TestAssertion(
                    description=f"Tool {tool_name} execution failed",
                    actual=str(e),
                    expected="Success",
                    operator="equals",
                    passed=False,
                    error_message=str(e),
                )
            )
            raise


class PerformanceTest(BaseTest):
    """Performance and load testing"""

    def __init__(self, test_id: str, name: str, **kwargs):
        super().__init__(test_id, name, TestType.PERFORMANCE, **kwargs)
        self.performance_metrics: Dict[str, Any] = {}

    def measure_execution_time(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, float]:
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        self.performance_metrics["execution_time"] = execution_time

        return result, execution_time

    def test_latency(self, func: Callable, max_latency: float, *args, **kwargs):
        """Test function latency"""
        result, execution_time = self.measure_execution_time(func, *args, **kwargs)

        self.assert_less_than(
            execution_time,
            max_latency,
            f"Execution time should be less than {max_latency}s",
        )

        return result

    def test_throughput(
        self, func: Callable, min_throughput: float, duration: float, *args, **kwargs
    ):
        """Test function throughput"""
        start_time = time.time()
        executions = 0

        while time.time() - start_time < duration:
            func(*args, **kwargs)
            executions += 1

        actual_throughput = executions / duration
        self.performance_metrics["throughput"] = actual_throughput

        self.assert_greater_than(
            actual_throughput,
            min_throughput,
            f"Throughput should be at least {min_throughput} ops/sec",
        )

        return actual_throughput


class LoadTest(BaseTest):
    """Load testing with concurrent execution"""

    def __init__(self, test_id: str, name: str, **kwargs):
        super().__init__(test_id, name, TestType.LOAD, **kwargs)
        self.load_metrics: Dict[str, Any] = {}

    def test_concurrent_load(
        self, func: Callable, concurrent_users: int, duration: float, *args, **kwargs
    ):
        """Test system under concurrent load"""

        executor = ThreadPoolExecutor(max_workers=concurrent_users)
        futures = []
        start_time = time.time()

        # Submit concurrent tasks
        for _ in range(concurrent_users):
            future = executor.submit(
                self._execute_load_worker, func, start_time, duration, *args, **kwargs
            )
            futures.append(future)

        # Collect results
        successful_executions = 0
        failed_executions = 0
        total_response_time = 0.0

        for future in as_completed(futures):
            try:
                executions, total_time = future.result()
                successful_executions += executions
                total_response_time += total_time
            except Exception as e:
                failed_executions += 1
                logging.error(f"Load test worker failed: {e}")

        # Calculate metrics
        total_executions = successful_executions + failed_executions
        success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0
        )
        avg_response_time = (
            total_response_time / successful_executions
            if successful_executions > 0
            else 0
        )
        throughput = total_executions / duration

        self.load_metrics.update(
            {
                "concurrent_users": concurrent_users,
                "duration": duration,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": success_rate,
                "average_response_time": avg_response_time,
                "throughput": throughput,
            }
        )

        executor.shutdown(wait=True)

        return self.load_metrics

    def _execute_load_worker(
        self, func: Callable, start_time: float, duration: float, *args, **kwargs
    ):
        """Worker function for load testing"""
        executions = 0
        total_time = 0.0

        while time.time() - start_time < duration:
            exec_start = time.time()
            try:
                func(*args, **kwargs)
                exec_end = time.time()
                total_time += exec_end - exec_start
                executions += 1
            except Exception as e:
                logging.error(f"Load test execution failed: {e}")
                break

            # Small delay to prevent overwhelming the system
            time.sleep(0.001)

        return executions, total_time


class TestRunner:
    """Executes tests and generates reports"""

    def __init__(self, artifacts_directory: Optional[str] = None):
        self.artifacts_directory = artifacts_directory or tempfile.mkdtemp(
            prefix="test_artifacts_"
        )
        self.test_results: List[TestResult] = []
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Create artifacts directory
        os.makedirs(self.artifacts_directory, exist_ok=True)

        logging.info(
            f"Test runner initialized with artifacts directory: {self.artifacts_directory}"
        )

    def run_test(self, test: BaseTest) -> TestResult:
        """Run a single test"""
        result = TestResult(
            test_id=test.test_id,
            test_name=test.name,
            test_type=test.test_type,
            status=TestStatus.PENDING,
            start_time=time.time(),
        )

        try:
            # Set status to running
            result.status = TestStatus.RUNNING

            # Run setup hooks
            for hook in test.setup_hooks:
                hook()

            # Execute test with timeout
            future = self.executor.submit(test.execute)
            test_result = future.result(timeout=test.timeout)

            # Check assertions
            failed_assertions = [a for a in test.assertions if not a.passed]

            if failed_assertions:
                result.status = TestStatus.FAILED
                result.error_message = f"{len(failed_assertions)} assertion(s) failed"
            else:
                result.status = TestStatus.PASSED

            result.assertions = test.assertions.copy()

        except TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {test.timeout} seconds"

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.stack_trace = traceback.format_exc()

        finally:
            # Run teardown hooks
            for hook in test.teardown_hooks:
                try:
                    hook()
                except Exception as e:
                    logging.error(f"Teardown hook failed: {e}")

            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            # Add performance metrics if available
            if hasattr(test, "performance_metrics"):
                result.metadata["performance_metrics"] = test.performance_metrics
            if hasattr(test, "load_metrics"):
                result.metadata["load_metrics"] = test.load_metrics

        self.test_results.append(result)
        return result

    def run_test_suite(
        self, test_suite: TestSuite, parallel: bool = False
    ) -> TestReport:
        """Run a complete test suite"""

        execution_id = str(uuid.uuid4())
        report = TestReport(
            suite_name=test_suite.name,
            execution_id=execution_id,
            start_time=time.time(),
            total_tests=len(test_suite.tests),
            artifacts_directory=self.artifacts_directory,
        )

        try:
            # Run suite setup hooks
            for hook in test_suite.setup_hooks:
                hook()

            # Execute tests
            if parallel:
                futures = []
                for test in test_suite.tests:
                    future = self.executor.submit(self.run_test, test)
                    futures.append(future)

                for future in as_completed(futures):
                    result = future.result()
                    report.test_results.append(result)
            else:
                for test in test_suite.tests:
                    result = self.run_test(test)
                    report.test_results.append(result)

            # Calculate statistics
            for result in report.test_results:
                if result.status == TestStatus.PASSED:
                    report.passed_tests += 1
                elif result.status == TestStatus.FAILED:
                    report.failed_tests += 1
                elif result.status == TestStatus.SKIPPED:
                    report.skipped_tests += 1
                else:
                    report.error_tests += 1

        except Exception as e:
            logging.error(f"Test suite execution failed: {e}")

        finally:
            # Run suite teardown hooks
            for hook in test_suite.teardown_hooks:
                try:
                    hook()
                except Exception as e:
                    logging.error(f"Suite teardown hook failed: {e}")

            report.end_time = time.time()

        # Generate report artifacts
        self._generate_report_artifacts(report)

        return report

    def _generate_report_artifacts(self, report: TestReport):
        """Generate test report artifacts"""

        # JSON report
        json_report_path = os.path.join(
            self.artifacts_directory, f"test_report_{report.execution_id}.json"
        )
        with open(json_report_path, "w") as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = self._dataclass_to_dict(report)
            json.dump(report_dict, f, indent=2, default=str)

        # HTML report
        html_report_path = os.path.join(
            self.artifacts_directory, f"test_report_{report.execution_id}.html"
        )
        self._generate_html_report(report, html_report_path)

        # CSV summary
        csv_report_path = os.path.join(
            self.artifacts_directory, f"test_summary_{report.execution_id}.csv"
        )
        self._generate_csv_summary(report, csv_report_path)

        logging.info(f"Test reports generated in {self.artifacts_directory}")

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Convert dataclass to dictionary recursively"""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = self._dataclass_to_dict(value)
            return result
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    def _generate_html_report(self, report: TestReport, file_path: str):
        """Generate HTML test report"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {report.suite_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        .skipped {{ color: gray; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .test-details {{ margin-top: 10px; }}
        .assertion {{ margin-left: 20px; }}
    </style>
</head>
<body>
    <h1>Test Report: {report.suite_name}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Execution ID:</strong> {report.execution_id}</p>
        <p><strong>Start Time:</strong> {time.ctime(report.start_time)}</p>
        <p><strong>End Time:</strong> {time.ctime(report.end_time) if report.end_time else 'N/A'}</p>
        <p><strong>Total Tests:</strong> {report.total_tests}</p>
        <p><strong class="passed">Passed:</strong> {report.passed_tests}</p>
        <p><strong class="failed">Failed:</strong> {report.failed_tests}</p>
        <p><strong class="error">Errors:</strong> {report.error_tests}</p>
        <p><strong class="skipped">Skipped:</strong> {report.skipped_tests}</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Type</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Assertions</th>
        </tr>
"""

        for result in report.test_results:
            status_class = result.status.value
            duration_str = f"{result.duration:.3f}s" if result.duration else "N/A"
            assertions_count = len(result.assertions)

            html_content += f"""
        <tr>
            <td>{result.test_name}</td>
            <td>{result.test_type.value}</td>
            <td class="{status_class}">{result.status.value.upper()}</td>
            <td>{duration_str}</td>
            <td>{assertions_count}</td>
        </tr>
"""

        html_content += """
    </table>
    
    <h2>Detailed Results</h2>
"""

        for result in report.test_results:
            html_content += f"""
    <div class="test-details">
        <h3>{result.test_name} - {result.status.value.upper()}</h3>
        <p><strong>Type:</strong> {result.test_type.value}</p>
        <p><strong>Duration:</strong> {f"{result.duration:.3f}s" if result.duration else "N/A"}</p>
"""

            if result.error_message:
                html_content += f"<p><strong>Error:</strong> {result.error_message}</p>"

            if result.assertions:
                html_content += "<h4>Assertions:</h4>"
                for assertion in result.assertions:
                    status_class = "passed" if assertion.passed else "failed"
                    html_content += f'<div class="assertion {status_class}">'
                    html_content += f"<strong>{assertion.description}:</strong> "
                    html_content += f"{'PASSED' if assertion.passed else 'FAILED'}"
                    if assertion.error_message:
                        html_content += f" - {assertion.error_message}"
                    html_content += "</div>"

            html_content += "</div>"

        html_content += """
</body>
</html>
"""

        with open(file_path, "w") as f:
            f.write(html_content)

    def _generate_csv_summary(self, report: TestReport, file_path: str):
        """Generate CSV summary of test results"""

        import csv

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Test Name",
                    "Type",
                    "Status",
                    "Duration",
                    "Assertions",
                    "Error Message",
                ]
            )

            for result in report.test_results:
                writer.writerow(
                    [
                        result.test_name,
                        result.test_type.value,
                        result.status.value,
                        result.duration or 0,
                        len(result.assertions),
                        result.error_message or "",
                    ]
                )


# Example test implementations
class SampleAgentTest(BaseTest):
    """Sample agent test implementation"""

    def execute(self):
        """Execute sample agent test"""
        # Simulate agent testing
        time.sleep(0.1)  # Simulate some work

        # Test agent response time
        start_time = time.time()
        # Simulate agent call
        time.sleep(0.05)
        response_time = time.time() - start_time

        self.assert_less_than(
            response_time, 0.1, "Agent response time should be < 100ms"
        )

        # Test agent capability
        mock_result = {"status": "success", "data": "test_data"}
        self.assert_equals(
            mock_result["status"], "success", "Agent should return success status"
        )
        self.assert_contains(
            mock_result, "data", "Agent result should contain data field"
        )


class SampleToolTest(BaseTest):
    """Sample tool test implementation"""

    def execute(self):
        """Execute sample tool test"""
        # Simulate tool testing
        time.sleep(0.05)

        # Test tool execution
        mock_input = {"query": "test", "limit": 5}
        mock_output = {"results": ["item1", "item2"], "count": 2}

        self.assert_equals(
            len(mock_output["results"]), 2, "Tool should return 2 results"
        )
        self.assert_equals(mock_output["count"], 2, "Count should match results length")

        # Test error handling
        def failing_function():
            raise ValueError("Test error")

        self.assert_raises(ValueError, failing_function)


class SamplePerformanceTest(PerformanceTest):
    """Sample performance test implementation"""

    def execute(self):
        """Execute sample performance test"""

        def test_function():
            # Simulate some processing
            time.sleep(0.01)
            return "result"

        # Test latency
        result = self.test_latency(test_function, 0.05)  # Max 50ms
        self.assert_equals(result, "result", "Function should return expected result")

        # Test throughput
        throughput = self.test_throughput(
            test_function, 50, 1.0
        )  # Min 50 ops/sec for 1 second
        self.assert_greater_than(
            throughput, 50, "Throughput should be at least 50 ops/sec"
        )


# Factory functions for creating test suites
def create_agent_test_suite(agent_system) -> TestSuite:
    """Create test suite for agent system"""

    suite = TestSuite(
        name="Agent System Tests",
        description="Comprehensive tests for agent system functionality",
        tags=["agents", "integration"],
    )

    # Add various agent tests
    suite.tests.extend(
        [
            SampleAgentTest(
                "agent_001", "Basic Agent Communication", TestType.INTEGRATION
            ),
            SampleAgentTest("agent_002", "Agent Error Handling", TestType.INTEGRATION),
            SampleAgentTest("agent_003", "Agent Performance", TestType.PERFORMANCE),
        ]
    )

    return suite


def create_tool_test_suite(tool_registry) -> TestSuite:
    """Create test suite for tool system"""

    suite = TestSuite(
        name="Tool System Tests",
        description="Comprehensive tests for tool system functionality",
        tags=["tools", "unit"],
    )

    # Add various tool tests
    suite.tests.extend(
        [
            SampleToolTest("tool_001", "Basic Tool Execution", TestType.UNIT),
            SampleToolTest("tool_002", "Tool Error Handling", TestType.UNIT),
            SampleToolTest("tool_003", "Tool Integration", TestType.INTEGRATION),
        ]
    )

    return suite


def create_performance_test_suite() -> TestSuite:
    """Create performance test suite"""

    suite = TestSuite(
        name="Performance Tests",
        description="Performance and load tests for the system",
        tags=["performance", "load"],
    )

    # Add performance tests
    suite.tests.extend(
        [
            SamplePerformanceTest("perf_001", "Response Time Test"),
            SamplePerformanceTest("perf_002", "Throughput Test"),
            SamplePerformanceTest("perf_003", "Scalability Test"),
        ]
    )

    return suite


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create test runner
    runner = TestRunner()

    # Create and run test suites
    suites = [
        create_agent_test_suite(None),  # Would pass actual agent system
        create_tool_test_suite(None),  # Would pass actual tool registry
        create_performance_test_suite(),
    ]

    all_reports = []

    for suite in suites:
        print(f"\nRunning test suite: {suite.name}")
        report = runner.run_test_suite(suite, parallel=False)
        all_reports.append(report)

        print(f"Suite Results:")
        print(f"  Total Tests: {report.total_tests}")
        print(f"  Passed: {report.passed_tests}")
        print(f"  Failed: {report.failed_tests}")
        print(f"  Errors: {report.error_tests}")
        print(f"  Skipped: {report.skipped_tests}")

        if report.failed_tests > 0 or report.error_tests > 0:
            print(f"  Failed/Error Tests:")
            for result in report.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"    - {result.test_name}: {result.error_message}")

    print(f"\nTest artifacts saved to: {runner.artifacts_directory}")

    # Calculate overall statistics
    total_tests = sum(r.total_tests for r in all_reports)
    total_passed = sum(r.passed_tests for r in all_reports)
    total_failed = sum(r.failed_tests for r in all_reports)
    total_errors = sum(r.error_tests for r in all_reports)

    print(f"\nOverall Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"  Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    print(f"  Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)")
