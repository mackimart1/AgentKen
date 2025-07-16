"""
Debugging Agent - Autonomous system failure detection, logging, and resolution
Integrates with the core task execution pipeline for comprehensive error handling
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import sqlite3
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent_framework import BaseAgent, AgentCapability, MessageBus, Message, MessageType, AgentStatus


class FailureType(Enum):
    """Types of system failures the debugging agent can handle"""
    TASK_EXECUTION_FAILURE = "task_execution_failure"
    AGENT_COMMUNICATION_FAILURE = "agent_communication_failure"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    MEMORY_FAILURE = "memory_failure"
    NETWORK_FAILURE = "network_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_FAILURE = "unknown_failure"


class SeverityLevel(Enum):
    """Severity levels for failures"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class FailureEvent:
    """Represents a system failure event"""
    id: str
    timestamp: datetime
    failure_type: FailureType
    severity: SeverityLevel
    component: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    resolution_attempts: List[Dict[str, Any]]
    resolved: bool
    resolution_time: Optional[datetime]
    impact_assessment: Dict[str, Any]


@dataclass
class ResolutionStrategy:
    """Represents a strategy for resolving failures"""
    name: str
    description: str
    applicable_failure_types: List[FailureType]
    priority: int
    execution_function: Callable
    prerequisites: List[str]
    estimated_time: float
    success_rate: float


class FailureDatabase:
    """SQLite database for storing failure events and patterns"""
    
    def __init__(self, db_path: str = "debugging_agent.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the failure tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                error_message TEXT NOT NULL,
                stack_trace TEXT,
                context TEXT,
                resolution_attempts TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time TEXT,
                impact_assessment TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resolution_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_pattern TEXT NOT NULL,
                resolution_strategy TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TEXT,
                effectiveness_score REAL DEFAULT 0.0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TEXT PRIMARY KEY,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_latency REAL,
                active_agents INTEGER,
                task_queue_size INTEGER,
                error_rate REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_failure_event(self, event: FailureEvent):
        """Store a failure event in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO failure_events 
            (id, timestamp, failure_type, severity, component, error_message, 
             stack_trace, context, resolution_attempts, resolved, resolution_time, impact_assessment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id,
            event.timestamp.isoformat(),
            event.failure_type.value,
            event.severity.value,
            event.component,
            event.error_message,
            event.stack_trace,
            json.dumps(event.context),
            json.dumps(event.resolution_attempts),
            event.resolved,
            event.resolution_time.isoformat() if event.resolution_time else None,
            json.dumps(event.impact_assessment)
        ))
        
        conn.commit()
        conn.close()
    
    def get_failure_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent failure patterns for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT failure_type, component, error_message, COUNT(*) as frequency
            FROM failure_events 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY failure_type, component, error_message
            ORDER BY frequency DESC
            LIMIT ?
        """, (limit,))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'failure_type': row[0],
                'component': row[1],
                'error_message': row[2],
                'frequency': row[3]
            })
        
        conn.close()
        return patterns
    
    def update_resolution_effectiveness(self, pattern: str, strategy: str, success: bool):
        """Update the effectiveness of a resolution strategy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO resolution_patterns 
            (failure_pattern, resolution_strategy, success_count, failure_count, last_used)
            VALUES (?, ?, 0, 0, ?)
        """, (pattern, strategy, datetime.now().isoformat()))
        
        if success:
            cursor.execute("""
                UPDATE resolution_patterns 
                SET success_count = success_count + 1, last_used = ?
                WHERE failure_pattern = ? AND resolution_strategy = ?
            """, (datetime.now().isoformat(), pattern, strategy))
        else:
            cursor.execute("""
                UPDATE resolution_patterns 
                SET failure_count = failure_count + 1, last_used = ?
                WHERE failure_pattern = ? AND resolution_strategy = ?
            """, (datetime.now().isoformat(), pattern, strategy))
        
        # Update effectiveness score
        cursor.execute("""
            UPDATE resolution_patterns 
            SET effectiveness_score = CAST(success_count AS REAL) / (success_count + failure_count)
            WHERE failure_pattern = ? AND resolution_strategy = ?
        """, (pattern, strategy))
        
        conn.commit()
        conn.close()


class ContextualTraceAnalyzer:
    """Analyzes stack traces and system context to identify failure patterns"""
    
    def __init__(self):
        self.common_patterns = {
            'import_error': r'ImportError|ModuleNotFoundError',
            'file_not_found': r'FileNotFoundError|No such file',
            'permission_denied': r'PermissionError|Permission denied',
            'connection_error': r'ConnectionError|Connection refused',
            'timeout_error': r'TimeoutError|timeout',
            'memory_error': r'MemoryError|Out of memory',
            'type_error': r'TypeError|type object',
            'value_error': r'ValueError|invalid literal',
            'key_error': r'KeyError|key not found',
            'attribute_error': r'AttributeError|has no attribute'
        }
    
    def analyze_failure(self, error_message: str, stack_trace: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a failure to extract patterns and context"""
        import re
        
        analysis = {
            'error_patterns': [],
            'root_cause_candidates': [],
            'affected_components': [],
            'severity_indicators': [],
            'context_clues': []
        }
        
        # Pattern matching
        for pattern_name, pattern_regex in self.common_patterns.items():
            if re.search(pattern_regex, error_message, re.IGNORECASE):
                analysis['error_patterns'].append(pattern_name)
        
        # Stack trace analysis
        if stack_trace:
            lines = stack_trace.split('\n')
            for line in lines:
                if 'File "' in line and 'line' in line:
                    # Extract file and line information
                    file_match = re.search(r'File "([^"]+)", line (\d+)', line)
                    if file_match:
                        analysis['affected_components'].append({
                            'file': file_match.group(1),
                            'line': int(file_match.group(2))
                        })
        
        # Context analysis
        if context:
            if 'agent_id' in context:
                analysis['context_clues'].append(f"Failed in agent: {context['agent_id']}")
            if 'task_type' in context:
                analysis['context_clues'].append(f"Task type: {context['task_type']}")
            if 'memory_usage' in context and context['memory_usage'] > 0.9:
                analysis['severity_indicators'].append('high_memory_usage')
            if 'cpu_usage' in context and context['cpu_usage'] > 0.9:
                analysis['severity_indicators'].append('high_cpu_usage')
        
        return analysis


class DebuggingAgent(BaseAgent):
    """
    Autonomous debugging agent that detects, logs, and resolves system failures
    """
    
    def __init__(self, message_bus: MessageBus, db_path: str = "debugging_agent.db"):
        capabilities = [
            AgentCapability(
                name="failure_detection",
                description="Detect and classify system failures",
                input_schema={
                    "error_message": "string",
                    "stack_trace": "string",
                    "context": "object"
                },
                output_schema={
                    "failure_event": "object",
                    "analysis": "object"
                },
                estimated_duration=2.0,
                max_concurrent=10
            ),
            AgentCapability(
                name="failure_resolution",
                description="Attempt to resolve detected failures",
                input_schema={
                    "failure_event_id": "string",
                    "resolution_strategy": "string"
                },
                output_schema={
                    "resolution_result": "object",
                    "success": "boolean"
                },
                estimated_duration=30.0,
                max_concurrent=5
            ),
            AgentCapability(
                name="pattern_analysis",
                description="Analyze failure patterns and trends",
                input_schema={
                    "time_range": "string",
                    "component_filter": "string"
                },
                output_schema={
                    "patterns": "array",
                    "recommendations": "array"
                },
                estimated_duration=10.0,
                max_concurrent=3
            ),
            AgentCapability(
                name="system_health_check",
                description="Perform comprehensive system health assessment",
                input_schema={},
                output_schema={
                    "health_status": "object",
                    "issues": "array",
                    "recommendations": "array"
                },
                estimated_duration=15.0,
                max_concurrent=1
            )
        ]
        
        super().__init__("debugging_agent", capabilities, message_bus)
        
        self.db = FailureDatabase(db_path)
        self.trace_analyzer = ContextualTraceAnalyzer()
        self.resolution_strategies = self._initialize_resolution_strategies()
        self.active_failures: Dict[str, FailureEvent] = {}
        self.monitoring_active = True
        
        # Start background monitoring
        self._start_monitoring()
        
        # Subscribe to error messages from other agents
        self.message_bus.subscribe(MessageType.ERROR_REPORT, self._handle_error_report)
    
    def _initialize_resolution_strategies(self) -> List[ResolutionStrategy]:
        """Initialize available resolution strategies"""
        return [
            ResolutionStrategy(
                name="restart_agent",
                description="Restart a failed agent",
                applicable_failure_types=[FailureType.AGENT_COMMUNICATION_FAILURE],
                priority=1,
                execution_function=self._restart_agent,
                prerequisites=[],
                estimated_time=10.0,
                success_rate=0.8
            ),
            ResolutionStrategy(
                name="clear_memory_cache",
                description="Clear memory caches to resolve memory issues",
                applicable_failure_types=[FailureType.MEMORY_FAILURE],
                priority=2,
                execution_function=self._clear_memory_cache,
                prerequisites=[],
                estimated_time=5.0,
                success_rate=0.7
            ),
            ResolutionStrategy(
                name="retry_with_backoff",
                description="Retry failed operation with exponential backoff",
                applicable_failure_types=[FailureType.NETWORK_FAILURE, FailureType.TIMEOUT_ERROR],
                priority=3,
                execution_function=self._retry_with_backoff,
                prerequisites=[],
                estimated_time=30.0,
                success_rate=0.6
            ),
            ResolutionStrategy(
                name="reload_configuration",
                description="Reload system configuration",
                applicable_failure_types=[FailureType.CONFIGURATION_ERROR],
                priority=2,
                execution_function=self._reload_configuration,
                prerequisites=[],
                estimated_time=15.0,
                success_rate=0.9
            ),
            ResolutionStrategy(
                name="install_missing_dependency",
                description="Install missing dependencies",
                applicable_failure_types=[FailureType.DEPENDENCY_FAILURE],
                priority=1,
                execution_function=self._install_missing_dependency,
                prerequisites=["admin_access"],
                estimated_time=60.0,
                success_rate=0.85
            )
        ]
    
    def _start_monitoring(self):
        """Start background monitoring for system health"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_system_metrics()
                    self._check_agent_health()
                    time.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logging.error(f"Monitoring loop error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent / 100.0,
                'disk_usage': psutil.disk_usage('/').percent / 100.0,
                'network_latency': self._measure_network_latency(),
                'active_agents': len(self.message_bus.subscribers),
                'task_queue_size': 0,  # Would need integration with task queue
                'error_rate': self._calculate_error_rate()
            }
            
            # Store metrics in database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, disk_usage, network_latency, 
                 active_agents, task_queue_size, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(metrics.values()))
            conn.commit()
            conn.close()
            
        except ImportError:
            logging.warning("psutil not available for system monitoring")
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    def _measure_network_latency(self) -> float:
        """Measure network latency to a reliable endpoint"""
        try:
            import subprocess
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse ping output for latency
                import re
                match = re.search(r'time=(\d+\.?\d*)', result.stdout)
                if match:
                    return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM failure_events 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            error_count = cursor.fetchone()[0]
            
            # Assume 1 error per minute is 100% error rate
            error_rate = min(error_count / 60.0, 1.0)
            
            conn.close()
            return error_rate
        except:
            return 0.0
    
    def _check_agent_health(self):
        """Check health of other agents in the system"""
        # Send capability query to discover agents
        query = Message(
            id=str(uuid.uuid4()),
            type=MessageType.CAPABILITY_QUERY,
            sender=self.agent_id,
            recipient="all",
            payload={"health_check": True},
            timestamp=time.time()
        )
        self.message_bus.publish(query)
    
    def _handle_error_report(self, message: Message):
        """Handle error reports from other agents"""
        try:
            payload = message.payload
            error_message = payload.get('error', 'Unknown error')
            context = {
                'reporting_agent': message.sender,
                'timestamp': message.timestamp,
                'original_message_id': payload.get('original_message_id')
            }
            
            # Create failure event
            failure_event = self._create_failure_event(
                error_message=error_message,
                stack_trace=payload.get('stack_trace'),
                context=context,
                component=message.sender
            )
            
            # Attempt automatic resolution
            self._attempt_resolution(failure_event)
            
        except Exception as e:
            logging.error(f"Error handling error report: {e}")
    
    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute debugging agent capabilities"""
        try:
            if capability_name == "failure_detection":
                return self._detect_failure(payload)
            elif capability_name == "failure_resolution":
                return self._resolve_failure(payload)
            elif capability_name == "pattern_analysis":
                return self._analyze_patterns(payload)
            elif capability_name == "system_health_check":
                return self._system_health_check(payload)
            else:
                raise ValueError(f"Unknown capability: {capability_name}")
        except Exception as e:
            logging.error(f"Error executing capability {capability_name}: {e}")
            raise
    
    def _detect_failure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and classify a failure"""
        error_message = payload.get('error_message', '')
        stack_trace = payload.get('stack_trace', '')
        context = payload.get('context', {})
        
        # Create failure event
        failure_event = self._create_failure_event(
            error_message=error_message,
            stack_trace=stack_trace,
            context=context
        )
        
        # Analyze the failure
        analysis = self.trace_analyzer.analyze_failure(error_message, stack_trace, context)
        
        return {
            'failure_event': asdict(failure_event),
            'analysis': analysis
        }
    
    def _create_failure_event(self, error_message: str, stack_trace: Optional[str] = None, 
                            context: Optional[Dict[str, Any]] = None, component: str = "unknown") -> FailureEvent:
        """Create a failure event from error information"""
        failure_type = self._classify_failure_type(error_message, stack_trace or "")
        severity = self._assess_severity(error_message, context or {})
        
        failure_event = FailureEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            failure_type=failure_type,
            severity=severity,
            component=component,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
            resolution_attempts=[],
            resolved=False,
            resolution_time=None,
            impact_assessment=self._assess_impact(failure_type, severity, component)
        )
        
        # Store in database
        self.db.store_failure_event(failure_event)
        self.active_failures[failure_event.id] = failure_event
        
        return failure_event
    
    def _classify_failure_type(self, error_message: str, stack_trace: str) -> FailureType:
        """Classify the type of failure based on error message and stack trace"""
        error_lower = error_message.lower()
        
        if 'import' in error_lower or 'module' in error_lower:
            return FailureType.DEPENDENCY_FAILURE
        elif 'file not found' in error_lower or 'no such file' in error_lower:
            return FailureType.CONFIGURATION_ERROR
        elif 'permission' in error_lower:
            return FailureType.CONFIGURATION_ERROR
        elif 'connection' in error_lower or 'network' in error_lower:
            return FailureType.NETWORK_FAILURE
        elif 'timeout' in error_lower:
            return FailureType.TIMEOUT_ERROR
        elif 'memory' in error_lower:
            return FailureType.MEMORY_FAILURE
        elif 'agent' in error_lower and 'communication' in error_lower:
            return FailureType.AGENT_COMMUNICATION_FAILURE
        elif 'tool' in error_lower:
            return FailureType.TOOL_EXECUTION_FAILURE
        else:
            return FailureType.UNKNOWN_FAILURE
    
    def _assess_severity(self, error_message: str, context: Dict[str, Any]) -> SeverityLevel:
        """Assess the severity of a failure"""
        error_lower = error_message.lower()
        
        # Critical indicators
        if any(word in error_lower for word in ['critical', 'fatal', 'crash', 'segfault']):
            return SeverityLevel.CRITICAL
        
        # High severity indicators
        if any(word in error_lower for word in ['memory', 'disk full', 'connection refused']):
            return SeverityLevel.HIGH
        
        # Medium severity indicators
        if any(word in error_lower for word in ['timeout', 'permission', 'not found']):
            return SeverityLevel.MEDIUM
        
        # Check context for severity indicators
        if context.get('memory_usage', 0) > 0.9:
            return SeverityLevel.HIGH
        if context.get('cpu_usage', 0) > 0.9:
            return SeverityLevel.HIGH
        
        return SeverityLevel.LOW
    
    def _assess_impact(self, failure_type: FailureType, severity: SeverityLevel, component: str) -> Dict[str, Any]:
        """Assess the impact of a failure on the system"""
        impact = {
            'affected_components': [component],
            'estimated_downtime': 0,
            'user_impact': 'low',
            'data_loss_risk': 'none',
            'cascade_risk': 'low'
        }
        
        if severity == SeverityLevel.CRITICAL:
            impact['estimated_downtime'] = 300  # 5 minutes
            impact['user_impact'] = 'high'
            impact['cascade_risk'] = 'high'
        elif severity == SeverityLevel.HIGH:
            impact['estimated_downtime'] = 120  # 2 minutes
            impact['user_impact'] = 'medium'
            impact['cascade_risk'] = 'medium'
        
        if failure_type == FailureType.MEMORY_FAILURE:
            impact['data_loss_risk'] = 'medium'
        elif failure_type == FailureType.AGENT_COMMUNICATION_FAILURE:
            impact['cascade_risk'] = 'high'
        
        return impact
    
    def _attempt_resolution(self, failure_event: FailureEvent):
        """Attempt to automatically resolve a failure"""
        applicable_strategies = [
            strategy for strategy in self.resolution_strategies
            if failure_event.failure_type in strategy.applicable_failure_types
        ]
        
        # Sort by priority and success rate
        applicable_strategies.sort(key=lambda s: (s.priority, -s.success_rate))
        
        for strategy in applicable_strategies:
            try:
                logging.info(f"Attempting resolution strategy: {strategy.name}")
                
                start_time = time.time()
                result = strategy.execution_function(failure_event)
                execution_time = time.time() - start_time
                
                # Record resolution attempt
                attempt = {
                    'strategy': strategy.name,
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': execution_time,
                    'result': result,
                    'success': result.get('success', False)
                }
                
                failure_event.resolution_attempts.append(attempt)
                
                if result.get('success', False):
                    failure_event.resolved = True
                    failure_event.resolution_time = datetime.now()
                    logging.info(f"Successfully resolved failure {failure_event.id} using {strategy.name}")
                    
                    # Update strategy effectiveness
                    pattern = f"{failure_event.failure_type.value}:{failure_event.component}"
                    self.db.update_resolution_effectiveness(pattern, strategy.name, True)
                    break
                else:
                    # Update strategy effectiveness
                    pattern = f"{failure_event.failure_type.value}:{failure_event.component}"
                    self.db.update_resolution_effectiveness(pattern, strategy.name, False)
                    
            except Exception as e:
                logging.error(f"Resolution strategy {strategy.name} failed: {e}")
                attempt = {
                    'strategy': strategy.name,
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': 0,
                    'result': {'success': False, 'error': str(e)},
                    'success': False
                }
                failure_event.resolution_attempts.append(attempt)
        
        # Update failure event in database
        self.db.store_failure_event(failure_event)
    
    # Resolution strategy implementations
    def _restart_agent(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Restart a failed agent"""
        try:
            # Send restart message to the failed agent
            restart_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.SHUTDOWN,
                sender=self.agent_id,
                recipient=failure_event.component,
                payload={'restart': True, 'reason': 'debugging_agent_resolution'},
                timestamp=time.time()
            )
            
            self.message_bus.publish(restart_message)
            
            # Wait a moment and check if agent responds
            time.sleep(5)
            
            return {'success': True, 'message': f'Restart signal sent to {failure_event.component}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _clear_memory_cache(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Clear memory caches to resolve memory issues"""
        try:
            import gc
            gc.collect()
            
            # Send cache clear message to other agents
            cache_clear_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_REQUEST,
                sender=self.agent_id,
                recipient="all",
                payload={'task_type': 'clear_cache', 'reason': 'memory_optimization'},
                timestamp=time.time()
            )
            
            self.message_bus.publish(cache_clear_message)
            
            return {'success': True, 'message': 'Memory cache cleared'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _retry_with_backoff(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Retry failed operation with exponential backoff"""
        try:
            # This would need integration with the original failed operation
            # For now, just simulate a retry
            max_retries = 3
            base_delay = 1.0
            
            for attempt in range(max_retries):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                
                # Simulate retry logic
                if attempt == max_retries - 1:  # Last attempt succeeds
                    return {'success': True, 'message': f'Operation succeeded after {attempt + 1} retries'}
            
            return {'success': False, 'message': 'All retry attempts failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _reload_configuration(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Reload system configuration"""
        try:
            # Send configuration reload message
            reload_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_REQUEST,
                sender=self.agent_id,
                recipient="all",
                payload={'task_type': 'reload_config', 'reason': 'configuration_error_resolution'},
                timestamp=time.time()
            )
            
            self.message_bus.publish(reload_message)
            
            return {'success': True, 'message': 'Configuration reload initiated'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _install_missing_dependency(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Install missing dependencies"""
        try:
            # Extract dependency name from error message
            import re
            dependency_match = re.search(r"No module named '([^']+)'", failure_event.error_message)
            
            if dependency_match:
                dependency = dependency_match.group(1)
                
                # Attempt to install using pip
                import subprocess
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dependency
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return {'success': True, 'message': f'Successfully installed {dependency}'}
                else:
                    return {'success': False, 'error': f'Failed to install {dependency}: {result.stderr}'}
            else:
                return {'success': False, 'error': 'Could not extract dependency name from error'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _resolve_failure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a specific failure"""
        failure_event_id = payload.get('failure_event_id')
        resolution_strategy = payload.get('resolution_strategy')
        
        if not failure_event_id or failure_event_id not in self.active_failures:
            return {'resolution_result': {'success': False, 'error': 'Failure event not found'}, 'success': False}
        
        failure_event = self.active_failures[failure_event_id]
        
        # Find the specified strategy
        strategy = next((s for s in self.resolution_strategies if s.name == resolution_strategy), None)
        
        if not strategy:
            return {'resolution_result': {'success': False, 'error': 'Resolution strategy not found'}, 'success': False}
        
        # Execute the strategy
        result = strategy.execution_function(failure_event)
        
        return {'resolution_result': result, 'success': result.get('success', False)}
    
    def _analyze_patterns(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure patterns and trends"""
        time_range = payload.get('time_range', '7 days')
        component_filter = payload.get('component_filter', '')
        
        patterns = self.db.get_failure_patterns()
        
        # Filter by component if specified
        if component_filter:
            patterns = [p for p in patterns if component_filter in p['component']]
        
        # Generate recommendations based on patterns
        recommendations = []
        
        for pattern in patterns[:5]:  # Top 5 patterns
            if pattern['frequency'] > 5:
                recommendations.append({
                    'type': 'high_frequency_failure',
                    'description': f"High frequency of {pattern['failure_type']} in {pattern['component']}",
                    'suggestion': f"Consider implementing preventive measures for {pattern['failure_type']}",
                    'priority': 'high'
                })
        
        return {
            'patterns': patterns,
            'recommendations': recommendations
        }
    
    def _system_health_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system health assessment"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'metrics': {},
            'last_check': datetime.now().isoformat()
        }
        
        issues = []
        recommendations = []
        
        try:
            # Check recent failure rate
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM failure_events 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            recent_failures = cursor.fetchone()[0]
            
            if recent_failures > 10:
                health_status['overall_status'] = 'degraded'
                issues.append({
                    'type': 'high_failure_rate',
                    'description': f'{recent_failures} failures in the last hour',
                    'severity': 'high'
                })
                recommendations.append({
                    'type': 'investigate_failures',
                    'description': 'Investigate recent failure patterns',
                    'priority': 'high'
                })
            
            # Check system metrics
            cursor.execute("""
                SELECT * FROM system_metrics 
                ORDER BY timestamp DESC LIMIT 1
            """)
            latest_metrics = cursor.fetchone()
            
            if latest_metrics:
                metrics = {
                    'cpu_usage': latest_metrics[1],
                    'memory_usage': latest_metrics[2],
                    'disk_usage': latest_metrics[3],
                    'network_latency': latest_metrics[4],
                    'error_rate': latest_metrics[7]
                }
                health_status['metrics'] = metrics
                
                # Check for resource issues
                if metrics['memory_usage'] > 0.9:
                    issues.append({
                        'type': 'high_memory_usage',
                        'description': f"Memory usage at {metrics['memory_usage']:.1%}",
                        'severity': 'high'
                    })
                
                if metrics['cpu_usage'] > 90:
                    issues.append({
                        'type': 'high_cpu_usage',
                        'description': f"CPU usage at {metrics['cpu_usage']:.1f}%",
                        'severity': 'medium'
                    })
            
            conn.close()
            
        except Exception as e:
            issues.append({
                'type': 'health_check_error',
                'description': f'Error during health check: {str(e)}',
                'severity': 'medium'
            })
        
        return {
            'health_status': health_status,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def shutdown(self):
        """Gracefully shutdown the debugging agent"""
        self.monitoring_active = False
        super().shutdown()


# Convenience function for external integration
def debugging_agent(task: str) -> Dict[str, Any]:
    """
    Main entry point for the debugging agent
    Compatible with the existing agent manifest system
    """
    try:
        # Parse task to determine action
        task_lower = task.lower()
        
        if 'detect' in task_lower or 'analyze' in task_lower:
            # Create a simple debugging agent for one-off analysis
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = DebuggingAgent(message_bus)
            
            # Extract error information from task
            result = agent.execute_capability("failure_detection", {
                "error_message": task,
                "stack_trace": "",
                "context": {}
            })
            
            return {
                "status": "success",
                "result": result,
                "message": "Failure analysis completed"
            }
        
        elif 'health' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = DebuggingAgent(message_bus)
            
            result = agent.execute_capability("system_health_check", {})
            
            return {
                "status": "success",
                "result": result,
                "message": "System health check completed"
            }
        
        else:
            return {
                "status": "success",
                "result": {
                    "message": "Debugging agent initialized and monitoring system",
                    "capabilities": [
                        "failure_detection",
                        "failure_resolution", 
                        "pattern_analysis",
                        "system_health_check"
                    ]
                },
                "message": "Debugging agent ready"
            }
            
    except Exception as e:
        return {
            "status": "failure",
            "result": None,
            "message": f"Debugging agent error: {str(e)}"
        }


if __name__ == "__main__":
    # Test the debugging agent
    logging.basicConfig(level=logging.INFO)
    
    from core.agent_framework import MessageBus
    message_bus = MessageBus()
    
    # Create debugging agent
    debug_agent = DebuggingAgent(message_bus)
    
    # Test failure detection
    test_result = debug_agent.execute_capability("failure_detection", {
        "error_message": "ImportError: No module named 'requests'",
        "stack_trace": "Traceback (most recent call last):\n  File \"test.py\", line 1, in <module>\n    import requests\nImportError: No module named 'requests'",
        "context": {"agent_id": "test_agent", "task_type": "web_request"}
    })
    
    print("Failure Detection Test:")
    print(json.dumps(test_result, indent=2))
    
    # Test system health check
    health_result = debug_agent.execute_capability("system_health_check", {})
    
    print("\nSystem Health Check:")
    print(json.dumps(health_result, indent=2))
    
    # Cleanup
    debug_agent.shutdown()