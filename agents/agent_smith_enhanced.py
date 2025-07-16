"""
Enhanced Agent Smith: Advanced agent architect with self-healing, versioning, and testing.

Key Enhancements:
1. Self-Healing: Monitors for crashes/anomalies and autonomously recovers
2. Agent Versioning: Tracks changes and enables rollback to previous versions
3. Testing Framework: Automated validation before deployment

This enhanced version provides robust agent development with comprehensive quality assurance.
"""

from typing import Literal, Optional, List, Dict, Any, Tuple
import os
import json
import shutil
import hashlib
import logging
import traceback
import threading
import time
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import psutil
import signal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

import config
import utils
import memory_manager

# Setup enhanced logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize memory manager
try:
    memory_manager_instance = memory_manager.MemoryManager()
except Exception as e:
    logger.warning(f"Memory manager initialization failed: {e}")
    memory_manager_instance = None


class AgentVersion:
    """Represents a version of an agent with metadata."""
    
    def __init__(self, agent_name: str, version: str, file_path: str, 
                 created_at: datetime = None, metadata: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.version = version
        self.file_path = file_path
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA256 hash of the agent file."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {self.file_path}: {e}")
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentVersion':
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            version=data["version"],
            file_path=data["file_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


class TestResult(Enum):
    """Test result enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Represents a test case for an agent."""
    name: str
    description: str
    test_function: str
    expected_result: Any = None
    timeout: int = 30
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestSuite:
    """Collection of test cases for an agent."""
    agent_name: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code
        }


@dataclass
class HealthMetrics:
    """Health monitoring metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_error_rate(self) -> float:
        """Calculate error rate percentage."""
        total = self.error_count + self.success_count
        return (self.error_count / total * 100) if total > 0 else 0.0


class AgentVersionManager:
    """Manages agent versions and rollback capabilities."""
    
    def __init__(self, base_path: str = "agents"):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / ".versions"
        self.versions_path.mkdir(exist_ok=True)
        self.version_registry = self._load_version_registry()
    
    def _load_version_registry(self) -> Dict[str, List[AgentVersion]]:
        """Load version registry from disk."""
        registry_file = self.versions_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for agent_name, versions_data in data.items():
                        registry[agent_name] = [
                            AgentVersion.from_dict(v) for v in versions_data
                        ]
                    return registry
            except Exception as e:
                logger.error(f"Failed to load version registry: {e}")
        return {}
    
    def _save_version_registry(self):
        """Save version registry to disk."""
        registry_file = self.versions_path / "registry.json"
        try:
            data = {}
            for agent_name, versions in self.version_registry.items():
                data[agent_name] = [v.to_dict() for v in versions]
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version registry: {e}")
    
    def create_version(self, agent_name: str, file_path: str, 
                      metadata: Dict[str, Any] = None) -> AgentVersion:
        """Create a new version of an agent."""
        if agent_name not in self.version_registry:
            self.version_registry[agent_name] = []
        
        # Generate version number
        version_num = len(self.version_registry[agent_name]) + 1
        version = f"v{version_num}.0"
        
        # Create version directory
        version_dir = self.versions_path / agent_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy agent file to version directory
        version_file = version_dir / f"{agent_name}.py"
        if os.path.exists(file_path):
            shutil.copy2(file_path, version_file)
        
        # Create version object
        agent_version = AgentVersion(
            agent_name=agent_name,
            version=version,
            file_path=str(version_file),
            metadata=metadata or {}
        )
        
        # Add to registry
        self.version_registry[agent_name].append(agent_version)
        self._save_version_registry()
        
        logger.info(f"Created version {version} for agent {agent_name}")
        return agent_version
    
    def get_versions(self, agent_name: str) -> List[AgentVersion]:
        """Get all versions of an agent."""
        return self.version_registry.get(agent_name, [])
    
    def get_latest_version(self, agent_name: str) -> Optional[AgentVersion]:
        """Get the latest version of an agent."""
        versions = self.get_versions(agent_name)
        return versions[-1] if versions else None
    
    def rollback_to_version(self, agent_name: str, version: str) -> bool:
        """Rollback agent to a specific version."""
        try:
            # Find the target version
            target_version = None
            for v in self.get_versions(agent_name):
                if v.version == version:
                    target_version = v
                    break
            
            if not target_version:
                logger.error(f"Version {version} not found for agent {agent_name}")
                return False
            
            # Copy version file back to main location
            main_file = self.base_path / f"{agent_name}.py"
            if os.path.exists(target_version.file_path):
                shutil.copy2(target_version.file_path, main_file)
                logger.info(f"Rolled back {agent_name} to version {version}")
                return True
            else:
                logger.error(f"Version file not found: {target_version.file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback {agent_name} to {version}: {e}")
            return False
    
    def get_version_diff(self, agent_name: str, version1: str, version2: str) -> str:
        """Get diff between two versions."""
        try:
            v1 = next((v for v in self.get_versions(agent_name) if v.version == version1), None)
            v2 = next((v for v in self.get_versions(agent_name) if v.version == version2), None)
            
            if not v1 or not v2:
                return "One or both versions not found"
            
            # Simple diff implementation
            with open(v1.file_path, 'r') as f1, open(v2.file_path, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
            
            diff_lines = []
            max_lines = max(len(lines1), len(lines2))
            
            for i in range(max_lines):
                line1 = lines1[i] if i < len(lines1) else ""
                line2 = lines2[i] if i < len(lines2) else ""
                
                if line1 != line2:
                    diff_lines.append(f"Line {i+1}:")
                    diff_lines.append(f"  {version1}: {line1.strip()}")
                    diff_lines.append(f"  {version2}: {line2.strip()}")
            
            return "\n".join(diff_lines) if diff_lines else "No differences found"
            
        except Exception as e:
            return f"Error generating diff: {e}"


class AgentTestFramework:
    """Automated testing framework for agents."""
    
    def __init__(self, test_dir: str = "tests/agents"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
    
    def generate_test_suite(self, agent_name: str, agent_code: str) -> TestSuite:
        """Generate a comprehensive test suite for an agent."""
        test_suite = TestSuite(agent_name=agent_name)
        
        # Basic functionality test
        basic_test = TestCase(
            name="test_basic_functionality",
            description="Test basic agent functionality",
            test_function=f"""
def test_basic_functionality(self):
    \"\"\"Test that agent can be called and returns expected format.\"\"\"
    from agents.{agent_name} import {agent_name}
    
    result = {agent_name}("test task")
    
    self.assertIsInstance(result, dict)
    self.assertIn('status', result)
    self.assertIn('result', result)
    self.assertIn('message', result)
    self.assertIn(result['status'], ['success', 'failure'])
"""
        )
        test_suite.add_test_case(basic_test)
        
        # Error handling test
        error_test = TestCase(
            name="test_error_handling",
            description="Test agent error handling",
            test_function=f"""
def test_error_handling(self):
    \"\"\"Test that agent handles errors gracefully.\"\"\"
    from agents.{agent_name} import {agent_name}
    
    # Test with invalid input
    result = {agent_name}("")
    self.assertIsInstance(result, dict)
    
    # Test with None input
    try:
        result = {agent_name}(None)
        self.assertIsInstance(result, dict)
    except Exception:
        pass  # Expected for some agents
"""
        )
        test_suite.add_test_case(error_test)
        
        # Performance test
        performance_test = TestCase(
            name="test_performance",
            description="Test agent performance",
            test_function=f"""
def test_performance(self):
    \"\"\"Test that agent completes within reasonable time.\"\"\"
    import time
    from agents.{agent_name} import {agent_name}
    
    start_time = time.time()
    result = {agent_name}("simple test task")
    end_time = time.time()
    
    execution_time = end_time - start_time
    self.assertLess(execution_time, 30, "Agent took too long to execute")
""",
            timeout=35
        )
        test_suite.add_test_case(performance_test)
        
        # Integration test if agent uses tools
        if "from tools." in agent_code or "import tools." in agent_code:
            integration_test = TestCase(
                name="test_tool_integration",
                description="Test agent tool integration",
                test_function=f"""
def test_tool_integration(self):
    \"\"\"Test that agent integrates properly with tools.\"\"\"
    from agents.{agent_name} import {agent_name}
    
    # Test with a task that should use tools
    result = {agent_name}("test tool integration")
    
    # Should not crash and should return proper format
    self.assertIsInstance(result, dict)
    self.assertIn('status', result)
"""
            )
            test_suite.add_test_case(integration_test)
        
        return test_suite
    
    def create_test_file(self, test_suite: TestSuite) -> str:
        """Create a test file from a test suite."""
        test_file_path = self.test_dir / f"test_{test_suite.agent_name}.py"
        
        test_content = f'''"""
Test suite for {test_suite.agent_name} agent.
Generated automatically by Enhanced Agent Smith.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Test{test_suite.agent_name.title()}(unittest.TestCase):
    """Test cases for {test_suite.agent_name} agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        {test_suite.setup_code}
    
    def tearDown(self):
        """Clean up after tests."""
        {test_suite.teardown_code}
'''
        
        # Add test methods
        for test_case in test_suite.test_cases:
            test_content += f"\n    {test_case.test_function}\n"
        
        test_content += '''

if __name__ == '__main__':
    unittest.main()
'''
        
        # Write test file
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_file_path}")
        return str(test_file_path)
    
    def run_tests(self, agent_name: str) -> Dict[str, Any]:
        """Run tests for an agent and return results."""
        test_file = self.test_dir / f"test_{agent_name}.py"
        
        if not test_file.exists():
            return {
                "status": "error",
                "message": f"Test file not found: {test_file}",
                "results": {}
            }
        
        try:
            # Run tests using subprocess
            cmd = [
                "python", "-m", "unittest", 
                f"tests.agents.test_{agent_name}", "-v"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                cwd=os.getcwd()
            )
            
            # Parse results
            test_results = self._parse_test_output(result.stdout, result.stderr)
            
            overall_status = "success" if result.returncode == 0 else "failure"
            
            return {
                "status": overall_status,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results": test_results,
                "message": f"Tests completed with return code {result.returncode}"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "message": "Tests timed out after 120 seconds",
                "results": {}
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error running tests: {e}",
                "results": {}
            }
    
    def _parse_test_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse unittest output to extract test results."""
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "test_details": []
        }
        
        lines = stdout.split('\n') + stderr.split('\n')
        
        for line in lines:
            if "test_" in line and ("ok" in line or "FAIL" in line or "ERROR" in line):
                results["total_tests"] += 1
                
                if "ok" in line:
                    results["passed"] += 1
                    status = TestResult.PASSED
                elif "FAIL" in line:
                    results["failed"] += 1
                    status = TestResult.FAILED
                elif "ERROR" in line:
                    results["errors"] += 1
                    status = TestResult.ERROR
                else:
                    status = TestResult.SKIPPED
                
                results["test_details"].append({
                    "name": line.strip(),
                    "status": status.value
                })
        
        return results


class SelfHealingMonitor:
    """Self-healing monitoring system for Agent Smith."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.health_history = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.monitor_thread = None
        
        # Health thresholds
        self.cpu_threshold = 80.0  # CPU usage percentage
        self.memory_threshold = 80.0  # Memory usage percentage
        self.error_rate_threshold = 10.0  # Error rate percentage
        self.response_time_threshold = 30.0  # Response time in seconds
    
    def start_monitoring(self):
        """Start the self-healing monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Self-healing monitor started")
    
    def stop_monitoring(self):
        """Stop the self-healing monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Self-healing monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                health_status = self._assess_health(metrics)
                
                self.health_history.append((datetime.now(), health_status, metrics))
                
                # Keep only last 100 entries
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                if health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    self._attempt_recovery(health_status, metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get process-specific metrics if available
            try:
                process = psutil.Process()
                process_cpu = process.cpu_percent()
                process_memory = process.memory_percent()
            except:
                process_cpu = 0
                process_memory = 0
            
            return HealthMetrics(
                cpu_usage=max(cpu_percent, process_cpu),
                memory_usage=max(memory_percent, process_memory),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return HealthMetrics()
    
    def _assess_health(self, metrics: HealthMetrics) -> HealthStatus:
        """Assess health status based on metrics."""
        critical_conditions = 0
        warning_conditions = 0
        
        if metrics.cpu_usage > self.cpu_threshold:
            critical_conditions += 1
        elif metrics.cpu_usage > self.cpu_threshold * 0.8:
            warning_conditions += 1
        
        if metrics.memory_usage > self.memory_threshold:
            critical_conditions += 1
        elif metrics.memory_usage > self.memory_threshold * 0.8:
            warning_conditions += 1
        
        if metrics.error_rate > self.error_rate_threshold:
            critical_conditions += 1
        elif metrics.error_rate > self.error_rate_threshold * 0.8:
            warning_conditions += 1
        
        if critical_conditions > 0:
            return HealthStatus.CRITICAL
        elif warning_conditions > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _attempt_recovery(self, health_status: HealthStatus, metrics: HealthMetrics):
        """Attempt to recover from unhealthy state."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.error("Maximum recovery attempts reached")
            return
        
        self.recovery_attempts += 1
        logger.warning(f"Attempting recovery #{self.recovery_attempts} for {health_status.value} status")
        
        try:
            if health_status == HealthStatus.CRITICAL:
                self._critical_recovery(metrics)
            elif health_status == HealthStatus.WARNING:
                self._warning_recovery(metrics)
            
            # Reset recovery attempts on successful recovery
            time.sleep(5)  # Wait before checking again
            new_metrics = self._collect_metrics()
            new_status = self._assess_health(new_metrics)
            
            if new_status == HealthStatus.HEALTHY:
                self.recovery_attempts = 0
                logger.info("Recovery successful")
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    def _critical_recovery(self, metrics: HealthMetrics):
        """Perform critical recovery actions."""
        logger.warning("Performing critical recovery actions")
        
        # Clear memory caches
        try:
            import gc
            gc.collect()
        except:
            pass
        
        # Reset global state if needed
        global creation_state
        if hasattr(creation_state, 'errors') and len(creation_state.errors) > 10:
            creation_state.errors = creation_state.errors[-5:]  # Keep only recent errors
        
        # Log critical state
        if memory_manager_instance:
            try:
                memory_manager_instance.add_memory(
                    key=f"critical_recovery_{datetime.now().timestamp()}",
                    value=json.dumps({
                        "metrics": asdict(metrics),
                        "recovery_attempt": self.recovery_attempts
                    }),
                    memory_type="system_health",
                    agent_name="agent_smith_enhanced"
                )
            except:
                pass
    
    def _warning_recovery(self, metrics: HealthMetrics):
        """Perform warning-level recovery actions."""
        logger.info("Performing warning-level recovery actions")
        
        # Light cleanup
        try:
            import gc
            gc.collect()
        except:
            pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.health_history:
            return {"status": "unknown", "message": "No health data available"}
        
        latest_timestamp, latest_status, latest_metrics = self.health_history[-1]
        
        return {
            "status": latest_status.value,
            "timestamp": latest_timestamp.isoformat(),
            "metrics": asdict(latest_metrics),
            "recovery_attempts": self.recovery_attempts,
            "monitoring": self.is_monitoring
        }


# Initialize enhanced components
version_manager = AgentVersionManager()
test_framework = AgentTestFramework()
health_monitor = SelfHealingMonitor()

# Start health monitoring
health_monitor.start_monitoring()


class EnhancedAgentCreationPhase(Enum):
    """Enhanced phases including versioning and testing."""
    PLANNING = "planning"
    TOOL_CREATION = "tool_creation"
    AGENT_WRITING = "agent_writing"
    VERSIONING = "versioning"
    TEST_GENERATION = "test_generation"
    TESTING = "testing"
    FORMATTING = "formatting"
    LINTING = "linting"
    FINAL_TESTING = "final_testing"
    DEPLOYMENT = "deployment"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class EnhancedAgentCreationState:
    """Enhanced state with versioning and testing tracking."""
    is_complete: bool = False
    agent_name: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    max_duration: int = 900  # 15 minutes
    max_inactivity: int = 180  # 3 minutes
    current_phase: EnhancedAgentCreationPhase = EnhancedAgentCreationPhase.PLANNING
    files_written: List[str] = field(default_factory=list)
    tools_requested: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Enhanced tracking
    current_version: Optional[AgentVersion] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    rollback_available: bool = False
    health_checks_passed: bool = False
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def advance_phase(self, new_phase: EnhancedAgentCreationPhase):
        """Advance to the next phase and update activity."""
        self.current_phase = new_phase
        self.update_activity()
        logger.info(f"Enhanced agent creation phase advanced to: {new_phase.value}")
    
    def add_error(self, error: str):
        """Add an error to the error list."""
        self.errors.append(f"[{self.current_phase.value}] {error}")
        logger.error(f"Enhanced agent creation error in {self.current_phase.value}: {error}")
    
    def check_timeout(self) -> tuple[bool, Optional[str]]:
        """Check if the agent creation has timed out."""
        current_time = datetime.now()
        total_duration = (current_time - self.start_time).total_seconds()
        inactivity_duration = (current_time - self.last_activity).total_seconds()
        
        if total_duration > self.max_duration:
            return True, f"Enhanced agent creation timed out after {self.max_duration} seconds"
        if inactivity_duration > self.max_inactivity:
            return True, f"Enhanced agent creation stalled after {self.max_inactivity} seconds of inactivity"
        return False, None


# Enhanced system prompt
enhanced_system_prompt = """You are Enhanced Agent Smith, an advanced ReAct agent architect with self-healing, versioning, and comprehensive testing capabilities.

ENHANCED CAPABILITIES:
1. **Self-Healing**: Monitor for anomalies and autonomously recover
2. **Agent Versioning**: Track changes and enable rollback capabilities  
3. **Testing Framework**: Automated validation before deployment

YOUR ENHANCED WORKFLOW (CRITICAL - Follow in Order):

1. PLANNING PHASE
   - Analyze task requirements thoroughly
   - Design agent architecture and identify capabilities
   - Plan versioning strategy and testing approach

2. TOOL CREATION PHASE (if needed)
   - Use `assign_agent_to_task` to request new tools from tool_maker
   - Wait for confirmation before proceeding

3. AGENT WRITING PHASE
   - Write agent implementation to `agents/agent_name.py`
   - Follow enhanced agent format requirements
   - Include comprehensive error handling and logging

4. VERSIONING PHASE
   - Create initial version using `create_agent_version`
   - Set up version tracking and rollback capabilities

5. TEST GENERATION PHASE
   - Generate comprehensive test suite using `generate_agent_tests`
   - Include unit tests, integration tests, and performance tests
   - Create test file in `tests/agents/test_agent_name.py`

6. INITIAL TESTING PHASE
   - Run generated tests using `run_agent_tests`
   - Verify all tests pass before proceeding
   - Fix any test failures and re-run

7. CODE QUALITY PHASE
   - Format code: `black agents/agent_name.py tests/agents/test_agent_name.py`
   - Lint code: `ruff check agents/agent_name.py tests/agents/test_agent_name.py --fix`
   - Fix any remaining linting issues

8. FINAL TESTING PHASE
   - Run complete test suite again using `run_agent_tests`
   - Perform integration testing with existing system
   - Validate performance benchmarks

9. DEPLOYMENT PHASE
   - Update agent manifest using `update_agent_manifest`
   - Perform final health checks
   - Enable monitoring for the new agent

10. COMPLETION PHASE
    - Generate success message only after ALL steps verified
    - Format: "Successfully created, versioned, tested, and deployed agent: 'agent_name'"

ENHANCED REQUIREMENTS:
- **ALWAYS** create versions before making changes
- **ALWAYS** run tests before proceeding to next phase
- **NEVER** deploy without passing all tests
- **ALWAYS** enable rollback capabilities
- **MONITOR** system health throughout process

ENHANCED AGENT FORMAT:
```python
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def agent_name(task: str) -> Dict[str, Any]:
    \"\"\"
    Enhanced agent with comprehensive error handling and monitoring.
    
    Args:
        task (str): The task to perform
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
    \"\"\"
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting {agent_name.__name__} with task: {task}")
        
        # Validate input
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")
        
        # Agent logic here
        result = "agent output"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed {agent_name.__name__} in {execution_time:.2f}s")
        
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully completed task: {task}",
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Error in {agent_name.__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__
        }
```

SELF-HEALING FEATURES:
- Monitor system resources during agent creation
- Automatically recover from memory/CPU issues
- Rollback to previous version if deployment fails
- Maintain health metrics and recovery logs

Use enhanced tools for versioning, testing, and monitoring throughout the process.
"""


# Initialize enhanced state
enhanced_creation_state = EnhancedAgentCreationState()


def agent_smith_enhanced(task: str) -> Dict[str, Any]:
    """
    Enhanced Agent Smith with self-healing, versioning, and testing capabilities.
    
    Args:
        task (str): The description of the agent to be created.
        
    Returns:
        Dict[str, Any]: Enhanced result with versioning and testing information
    """
    # Reset enhanced state
    global enhanced_creation_state
    enhanced_creation_state = EnhancedAgentCreationState()
    
    logger.info(f"Starting enhanced agent creation task: {task}")
    
    # Check system health before starting
    health_status = health_monitor.get_health_status()
    if health_status["status"] == "critical":
        return {
            "status": "failure",
            "result": None,
            "message": "System health critical - cannot start agent creation",
            "health_status": health_status
        }
    
    try:
        # Load enhanced tools
        enhanced_tools = _load_enhanced_tools()
        
        # Create enhanced workflow
        enhanced_workflow = _create_enhanced_workflow(enhanced_tools)
        
        # Execute enhanced workflow
        final_state = enhanced_workflow.invoke({
            "messages": [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=task)
            ]
        })
        
        # Extract results
        last_message_content = "No response generated"
        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                if isinstance(content, list):
                    last_message_content = next(
                        (item for item in content if isinstance(item, str)), 
                        str(content)
                    )
                else:
                    last_message_content = str(content)
        
        # Determine final status
        if (enhanced_creation_state.is_complete and 
            enhanced_creation_state.current_phase == EnhancedAgentCreationPhase.COMPLETE):
            status = "success"
            result_data = {
                "agent_name": enhanced_creation_state.agent_name,
                "version": enhanced_creation_state.current_version.version if enhanced_creation_state.current_version else None,
                "test_results": enhanced_creation_state.test_results,
                "files_created": enhanced_creation_state.files_written
            }
        else:
            status = "failure"
            result_data = None
        
        # Store enhanced memory
        _store_enhanced_memory(task, status, enhanced_creation_state)
        
        return {
            "status": status,
            "result": result_data,
            "message": last_message_content,
            "phase": enhanced_creation_state.current_phase.value,
            "files_created": enhanced_creation_state.files_written.copy(),
            "errors": enhanced_creation_state.errors.copy(),
            "test_results": enhanced_creation_state.test_results.copy(),
            "version_info": enhanced_creation_state.current_version.to_dict() if enhanced_creation_state.current_version else None,
            "health_status": health_monitor.get_health_status(),
            "rollback_available": enhanced_creation_state.rollback_available
        }
        
    except Exception as e:
        error_msg = f"Critical error in enhanced agent_smith: {str(e)}"
        logger.error(error_msg, exc_info=True)
        enhanced_creation_state.add_error(error_msg)
        
        # Attempt self-healing
        try:
            health_monitor._attempt_recovery(HealthStatus.CRITICAL, health_monitor._collect_metrics())
        except:
            pass
        
        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "phase": enhanced_creation_state.current_phase.value,
            "files_created": enhanced_creation_state.files_written.copy(),
            "errors": enhanced_creation_state.errors.copy(),
            "health_status": health_monitor.get_health_status(),
            "self_healing_attempted": True
        }


def _load_enhanced_tools() -> List[Any]:
    """Load enhanced tools including versioning and testing tools."""
    # Load base tools
    tools = utils.all_tool_functions()
    
    # Add enhanced tools (these would be implemented as separate tool files)
    enhanced_tool_names = [
        "create_agent_version",
        "rollback_agent_version", 
        "generate_agent_tests",
        "run_agent_tests",
        "update_agent_manifest",
        "check_system_health"
    ]
    
    logger.info(f"Loaded {len(tools)} tools for Enhanced Agent Smith")
    return tools


def _create_enhanced_workflow(tools: List[Any]):
    """Create the enhanced LangGraph workflow."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("reasoning", lambda state: _enhanced_reasoning_node(state, tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("reasoning")
    workflow.add_conditional_edges("reasoning", _enhanced_check_for_tool_calls)
    workflow.add_edge("tools", "reasoning")
    return workflow.compile()


def _enhanced_reasoning_node(state: MessagesState, tools: List[Any]) -> Dict[str, Any]:
    """Enhanced reasoning with health monitoring and recovery."""
    logger.info(f"Enhanced reasoning in phase: {enhanced_creation_state.current_phase.value}")
    
    # Check timeout and health
    is_timeout, timeout_message = enhanced_creation_state.check_timeout()
    if is_timeout:
        enhanced_creation_state.advance_phase(EnhancedAgentCreationPhase.FAILED)
        return {"messages": [AIMessage(content=f"Error: {timeout_message}")]}
    
    # Monitor system health
    health_status = health_monitor.get_health_status()
    if health_status["status"] == "critical":
        logger.warning("Critical health status detected during reasoning")
    
    enhanced_creation_state.update_activity()
    
    try:
        # Use Google Gemini for tool calling from hybrid configuration
        tool_model = config.get_model_for_tools()
        if tool_model is None:
            # Fallback to default model if hybrid setup fails
            tool_model = config.default_langchain_model
            logger.warning("Using fallback model for tools - may not support function calling")
        
        tooled_up_model = tool_model.bind_tools(tools)
        response = tooled_up_model.invoke(state["messages"])
        
        # Process enhanced response
        if isinstance(response, AIMessage):
            _process_enhanced_response(response)
        
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"Error in enhanced reasoning: {str(e)}"
        enhanced_creation_state.add_error(error_msg)
        logger.error(error_msg, exc_info=True)
        
        # Attempt recovery
        try:
            health_monitor._warning_recovery(health_monitor._collect_metrics())
        except:
            pass
        
        return {"messages": [AIMessage(content=f"Internal error: {error_msg}")]}


def _process_enhanced_response(response: AIMessage):
    """Process enhanced AI response with versioning and testing awareness."""
    content = response.content
    if isinstance(content, list):
        content = next((item for item in content if isinstance(item, str)), str(content))
    content = str(content)
    
    # Check for completion
    if "successfully created, versioned, tested, and deployed agent" in content.lower():
        enhanced_creation_state.advance_phase(EnhancedAgentCreationPhase.COMPLETE)
        enhanced_creation_state.is_complete = True
        
        # Extract agent name
        try:
            if "agent: '" in content:
                name_part = content.split("agent: '")[1]
                enhanced_creation_state.agent_name = name_part.split("'")[0]
        except (IndexError, AttributeError):
            logger.warning("Could not extract agent name from completion message")


def _enhanced_check_for_tool_calls(state: MessagesState) -> Literal["tools", "END"]:
    """Enhanced tool call checking with phase awareness."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Check completion states
    if (enhanced_creation_state.is_complete or 
        enhanced_creation_state.current_phase == EnhancedAgentCreationPhase.COMPLETE):
        logger.info(f"Enhanced agent creation completed: {enhanced_creation_state.agent_name}")
        return "END"
    
    if enhanced_creation_state.current_phase == EnhancedAgentCreationPhase.FAILED:
        logger.error("Enhanced agent creation failed")
        return "END"
    
    # Check for tool calls
    if (isinstance(last_message, AIMessage) and 
        hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return "tools"
    
    return "END"


def _store_enhanced_memory(task: str, status: str, state: EnhancedAgentCreationState):
    """Store enhanced memory with versioning and testing information."""
    if not memory_manager_instance:
        return
    
    try:
        memory_data = {
            "task": task,
            "status": status,
            "phase": state.current_phase.value,
            "files": state.files_written,
            "errors": state.errors,
            "test_results": state.test_results,
            "version_info": state.current_version.to_dict() if state.current_version else None,
            "health_checks_passed": state.health_checks_passed,
            "rollback_available": state.rollback_available
        }
        
        memory_manager_instance.add_memory(
            key=f"enhanced_agent_creation_{datetime.now().timestamp()}",
            value=json.dumps(memory_data),
            memory_type="enhanced_agent_creation_log",
            agent_name="agent_smith_enhanced"
        )
    except Exception as e:
        logger.warning(f"Failed to store enhanced memory: {e}")


# Cleanup function for graceful shutdown
def cleanup_enhanced_agent_smith():
    """Cleanup function for enhanced agent smith."""
    try:
        health_monitor.stop_monitoring()
        logger.info("Enhanced Agent Smith cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Register cleanup function
import atexit
atexit.register(cleanup_enhanced_agent_smith)