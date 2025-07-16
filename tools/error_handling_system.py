"""
Comprehensive Error Handling & Recovery System for AgentK

Key Features:
1. Centralized Error-Handling Agent: Logs exceptions, analyzes failure patterns, suggests fixes
2. Retry Logic with Exponential Backoff: Handles transient failures gracefully

This system provides enterprise-grade error handling and recovery capabilities.
"""

import asyncio
import functools
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    LOGIC = "logic"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class ErrorContext:
    """Context information for an error."""
    agent_id: str
    tool_name: str
    function_name: str
    parameters: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: str = "production"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ErrorRecord:
    """Comprehensive error record with analysis."""
    error_id: str
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: str
    retry_count: int = 0
    resolved: bool = False
    resolution_notes: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
            "suggested_fixes": self.suggested_fixes,
            "related_errors": self.related_errors,
            "metadata": self.metadata
        }


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: List[type] = field(default_factory=list)
    non_retryable_exceptions: List[type] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.retryable_exceptions:
            self.retryable_exceptions = [
                ConnectionError, TimeoutError, OSError,
                # Add more retryable exceptions as needed
            ]
        
        if not self.non_retryable_exceptions:
            self.non_retryable_exceptions = [
                ValueError, TypeError, AttributeError,
                PermissionError, FileNotFoundError
            ]


class ErrorPatternAnalyzer:
    """Analyzes error patterns and suggests fixes."""
    
    def __init__(self):
        self.pattern_rules = self._load_pattern_rules()
        self.fix_suggestions = self._load_fix_suggestions()
    
    def _load_pattern_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load error pattern recognition rules."""
        return {
            "connection_timeout": {
                "patterns": [
                    r"connection.*timeout",
                    r"timeout.*connection",
                    r"read timeout",
                    r"connect timeout"
                ],
                "category": ErrorCategory.TIMEOUT,
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF
            },
            "authentication_failed": {
                "patterns": [
                    r"authentication.*failed",
                    r"invalid.*credentials",
                    r"unauthorized",
                    r"401.*error",
                    r"access.*denied"
                ],
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.NO_RETRY
            },
            "permission_denied": {
                "patterns": [
                    r"permission.*denied",
                    r"access.*forbidden",
                    r"403.*error",
                    r"insufficient.*privileges"
                ],
                "category": ErrorCategory.PERMISSION,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.NO_RETRY
            },
            "resource_exhausted": {
                "patterns": [
                    r"out of memory",
                    r"disk.*full",
                    r"resource.*exhausted",
                    r"too many.*requests",
                    r"rate.*limit"
                ],
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.LINEAR_BACKOFF
            },
            "network_error": {
                "patterns": [
                    r"network.*error",
                    r"connection.*refused",
                    r"host.*unreachable",
                    r"dns.*resolution.*failed"
                ],
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF
            },
            "validation_error": {
                "patterns": [
                    r"validation.*failed",
                    r"invalid.*input",
                    r"schema.*error",
                    r"format.*error"
                ],
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": RetryStrategy.NO_RETRY
            },
            "configuration_error": {
                "patterns": [
                    r"configuration.*error",
                    r"config.*missing",
                    r"environment.*variable.*not.*set",
                    r"missing.*required.*parameter"
                ],
                "category": ErrorCategory.CONFIGURATION,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.NO_RETRY
            },
            "dependency_error": {
                "patterns": [
                    r"module.*not.*found",
                    r"import.*error",
                    r"dependency.*missing",
                    r"package.*not.*installed"
                ],
                "category": ErrorCategory.DEPENDENCY,
                "severity": ErrorSeverity.CRITICAL,
                "retry_strategy": RetryStrategy.NO_RETRY
            }
        }
    
    def _load_fix_suggestions(self) -> Dict[ErrorCategory, List[str]]:
        """Load fix suggestions for different error categories."""
        return {
            ErrorCategory.TIMEOUT: [
                "Increase timeout values in configuration",
                "Check network connectivity and latency",
                "Implement retry logic with exponential backoff",
                "Consider using connection pooling",
                "Monitor system load and performance"
            ],
            ErrorCategory.AUTHENTICATION: [
                "Verify API keys and credentials are correct",
                "Check if credentials have expired",
                "Ensure proper authentication headers are set",
                "Verify user permissions and access rights",
                "Check authentication service availability"
            ],
            ErrorCategory.PERMISSION: [
                "Verify user has required permissions",
                "Check file/directory access rights",
                "Ensure proper role assignments",
                "Review security policies and restrictions",
                "Contact administrator for access escalation"
            ],
            ErrorCategory.RESOURCE: [
                "Monitor and optimize memory usage",
                "Implement resource cleanup and garbage collection",
                "Scale up system resources if needed",
                "Implement rate limiting and throttling",
                "Optimize algorithms for better resource efficiency"
            ],
            ErrorCategory.NETWORK: [
                "Check network connectivity and DNS resolution",
                "Verify firewall and proxy settings",
                "Implement connection retry logic",
                "Use connection pooling and keep-alive",
                "Monitor network latency and bandwidth"
            ],
            ErrorCategory.VALIDATION: [
                "Validate input data format and schema",
                "Check required fields and data types",
                "Implement proper input sanitization",
                "Review API documentation for correct usage",
                "Add comprehensive input validation"
            ],
            ErrorCategory.CONFIGURATION: [
                "Check configuration file syntax and values",
                "Verify environment variables are set",
                "Review default configuration settings",
                "Validate configuration against schema",
                "Ensure all required parameters are provided"
            ],
            ErrorCategory.DEPENDENCY: [
                "Install missing packages and dependencies",
                "Check package versions and compatibility",
                "Update package manager and repositories",
                "Verify virtual environment setup",
                "Review import paths and module structure"
            ],
            ErrorCategory.LOGIC: [
                "Review algorithm logic and flow",
                "Check for edge cases and boundary conditions",
                "Implement proper error handling",
                "Add logging and debugging information",
                "Review code for potential race conditions"
            ],
            ErrorCategory.UNKNOWN: [
                "Enable detailed logging and debugging",
                "Reproduce error in controlled environment",
                "Check system logs and error messages",
                "Review recent changes and deployments",
                "Contact support with detailed error information"
            ]
        }
    
    def analyze_error(self, error_message: str, error_type: str, stack_trace: str) -> Tuple[ErrorCategory, ErrorSeverity, List[str]]:
        """Analyze error and return category, severity, and suggested fixes."""
        error_text = f"{error_type} {error_message} {stack_trace}".lower()
        
        # Try to match against known patterns
        for pattern_name, pattern_info in self.pattern_rules.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, error_text, re.IGNORECASE):
                    category = pattern_info["category"]
                    severity = pattern_info["severity"]
                    suggestions = self.fix_suggestions.get(category, [])
                    return category, severity, suggestions
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, self.fix_suggestions[ErrorCategory.UNKNOWN]
    
    def find_related_errors(self, error_record: ErrorRecord, error_history: List[ErrorRecord]) -> List[str]:
        """Find related errors based on patterns and context."""
        related = []
        
        for other_error in error_history:
            if other_error.error_id == error_record.error_id:
                continue
            
            # Check for similar error types
            if other_error.error_type == error_record.error_type:
                related.append(other_error.error_id)
                continue
            
            # Check for similar error messages
            if self._calculate_similarity(error_record.error_message, other_error.error_message) > 0.7:
                related.append(other_error.error_id)
                continue
            
            # Check for same context (agent, tool, function)
            if (other_error.context.agent_id == error_record.context.agent_id and
                other_error.context.tool_name == error_record.context.tool_name and
                other_error.context.function_name == error_record.context.function_name):
                related.append(other_error.error_id)
        
        return related[:10]  # Limit to 10 related errors
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple similarity calculation using common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class ErrorDatabase:
    """SQLite database for storing and querying error records."""
    
    def __init__(self, db_path: str = "error_handling.db"):
        self.db_path = db_path
        self._init_database()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """Initialize the error database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_records (
                    error_id TEXT PRIMARY KEY,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    user_id TEXT,
                    environment TEXT,
                    stack_trace TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_notes TEXT,
                    suggested_fixes TEXT,
                    related_errors TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_type ON error_records(error_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_category ON error_records(error_category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_tool ON error_records(agent_id, tool_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON error_records(timestamp)
            """)
            
            conn.commit()
    
    def store_error(self, error_record: ErrorRecord) -> bool:
        """Store an error record in the database."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO error_records (
                            error_id, error_type, error_message, error_category, severity,
                            agent_id, tool_name, function_name, parameters, timestamp,
                            session_id, user_id, environment, stack_trace, retry_count,
                            resolved, resolution_notes, suggested_fixes, related_errors, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        error_record.error_id,
                        error_record.error_type,
                        error_record.error_message,
                        error_record.error_category.value,
                        error_record.severity.value,
                        error_record.context.agent_id,
                        error_record.context.tool_name,
                        error_record.context.function_name,
                        json.dumps(error_record.context.parameters),
                        error_record.context.timestamp.isoformat(),
                        error_record.context.session_id,
                        error_record.context.user_id,
                        error_record.context.environment,
                        error_record.stack_trace,
                        error_record.retry_count,
                        error_record.resolved,
                        error_record.resolution_notes,
                        json.dumps(error_record.suggested_fixes),
                        json.dumps(error_record.related_errors),
                        json.dumps(error_record.metadata)
                    ))
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to store error record: {e}")
                return False
    
    def get_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Retrieve an error record by ID."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM error_records WHERE error_id = ?",
                        (error_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_error_record(row)
                    return None
            except Exception as e:
                logger.error(f"Failed to retrieve error record: {e}")
                return None
    
    def get_errors_by_pattern(self, agent_id: str = None, tool_name: str = None,
                             error_category: ErrorCategory = None, 
                             hours_back: int = 24, limit: int = 100) -> List[ErrorRecord]:
        """Get errors matching specific patterns."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    query = "SELECT * FROM error_records WHERE timestamp >= ?"
                    params = [datetime.now() - timedelta(hours=hours_back)]
                    
                    if agent_id:
                        query += " AND agent_id = ?"
                        params.append(agent_id)
                    
                    if tool_name:
                        query += " AND tool_name = ?"
                        params.append(tool_name)
                    
                    if error_category:
                        query += " AND error_category = ?"
                        params.append(error_category.value)
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    return [self._row_to_error_record(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to query error records: {e}")
                return []
    
    def get_error_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get error statistics for analysis."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    since = datetime.now() - timedelta(hours=hours_back)
                    
                    # Total errors
                    total_errors = conn.execute(
                        "SELECT COUNT(*) FROM error_records WHERE timestamp >= ?",
                        (since,)
                    ).fetchone()[0]
                    
                    # Errors by category
                    category_stats = {}
                    cursor = conn.execute("""
                        SELECT error_category, COUNT(*) as count 
                        FROM error_records 
                        WHERE timestamp >= ? 
                        GROUP BY error_category
                    """, (since,))
                    
                    for row in cursor:
                        category_stats[row[0]] = row[1]
                    
                    # Errors by severity
                    severity_stats = {}
                    cursor = conn.execute("""
                        SELECT severity, COUNT(*) as count 
                        FROM error_records 
                        WHERE timestamp >= ? 
                        GROUP BY severity
                    """, (since,))
                    
                    for row in cursor:
                        severity_stats[row[0]] = row[1]
                    
                    # Top error types
                    top_errors = []
                    cursor = conn.execute("""
                        SELECT error_type, COUNT(*) as count 
                        FROM error_records 
                        WHERE timestamp >= ? 
                        GROUP BY error_type 
                        ORDER BY count DESC 
                        LIMIT 10
                    """, (since,))
                    
                    for row in cursor:
                        top_errors.append({"error_type": row[0], "count": row[1]})
                    
                    # Resolution rate
                    resolved_count = conn.execute(
                        "SELECT COUNT(*) FROM error_records WHERE timestamp >= ? AND resolved = TRUE",
                        (since,)
                    ).fetchone()[0]
                    
                    resolution_rate = (resolved_count / total_errors * 100) if total_errors > 0 else 0
                    
                    return {
                        "total_errors": total_errors,
                        "resolved_errors": resolved_count,
                        "resolution_rate": resolution_rate,
                        "category_distribution": category_stats,
                        "severity_distribution": severity_stats,
                        "top_error_types": top_errors,
                        "time_period_hours": hours_back
                    }
            except Exception as e:
                logger.error(f"Failed to get error statistics: {e}")
                return {}
    
    def _row_to_error_record(self, row: sqlite3.Row) -> ErrorRecord:
        """Convert database row to ErrorRecord object."""
        context = ErrorContext(
            agent_id=row["agent_id"],
            tool_name=row["tool_name"],
            function_name=row["function_name"],
            parameters=json.loads(row["parameters"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            session_id=row["session_id"],
            user_id=row["user_id"],
            environment=row["environment"]
        )
        
        return ErrorRecord(
            error_id=row["error_id"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            error_category=ErrorCategory(row["error_category"]),
            severity=ErrorSeverity(row["severity"]),
            context=context,
            stack_trace=row["stack_trace"],
            retry_count=row["retry_count"],
            resolved=bool(row["resolved"]),
            resolution_notes=row["resolution_notes"],
            suggested_fixes=json.loads(row["suggested_fixes"]) if row["suggested_fixes"] else [],
            related_errors=json.loads(row["related_errors"]) if row["related_errors"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, default_config: RetryConfig = None):
        self.default_config = default_config or RetryConfig()
        self.retry_stats = defaultdict(int)
        self._lock = threading.RLock()
    
    def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        else:  # IMMEDIATE
            delay = 0
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def should_retry(self, exception: Exception, attempt: int, config: RetryConfig) -> bool:
        """Determine if an exception should be retried."""
        if attempt >= config.max_attempts:
            return False
        
        exception_type = type(exception)
        
        # Check non-retryable exceptions first
        for non_retryable in config.non_retryable_exceptions:
            if issubclass(exception_type, non_retryable):
                return False
        
        # Check retryable exceptions
        for retryable in config.retryable_exceptions:
            if issubclass(exception_type, retryable):
                return True
        
        # Default behavior based on exception type
        return self._is_retryable_by_default(exception)
    
    def _is_retryable_by_default(self, exception: Exception) -> bool:
        """Default retry logic for common exception types."""
        retryable_types = (
            ConnectionError, TimeoutError, OSError,
            # Add more as needed
        )
        
        non_retryable_types = (
            ValueError, TypeError, AttributeError,
            PermissionError, FileNotFoundError,
            KeyError, IndexError
        )
        
        if isinstance(exception, non_retryable_types):
            return False
        
        if isinstance(exception, retryable_types):
            return True
        
        # For unknown exceptions, be conservative
        return False
    
    def record_retry_attempt(self, function_name: str, attempt: int, success: bool):
        """Record retry attempt statistics."""
        with self._lock:
            self.retry_stats[f"{function_name}_attempts"] += 1
            if success:
                self.retry_stats[f"{function_name}_success_after_{attempt}"] += 1
            else:
                self.retry_stats[f"{function_name}_failed_after_{attempt}"] += 1
    
    def get_retry_stats(self) -> Dict[str, int]:
        """Get retry statistics."""
        with self._lock:
            return dict(self.retry_stats)


class CentralizedErrorHandler:
    """Centralized error handling agent for the system."""
    
    def __init__(self, db_path: str = "error_handling.db"):
        self.database = ErrorDatabase(db_path)
        self.pattern_analyzer = ErrorPatternAnalyzer()
        self.retry_manager = RetryManager()
        self.error_cache = deque(maxlen=1000)  # Recent errors cache
        self._lock = threading.RLock()
        
        # Start background analysis task
        self._start_background_analysis()
    
    def handle_error(self, exception: Exception, context: ErrorContext) -> ErrorRecord:
        """Handle an error and create comprehensive error record."""
        error_id = str(uuid.uuid4())
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        # Analyze error pattern
        category, severity, suggested_fixes = self.pattern_analyzer.analyze_error(
            error_message, error_type, stack_trace
        )
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            error_type=error_type,
            error_message=error_message,
            error_category=category,
            severity=severity,
            context=context,
            stack_trace=stack_trace,
            suggested_fixes=suggested_fixes
        )
        
        # Find related errors
        recent_errors = list(self.error_cache)
        error_record.related_errors = self.pattern_analyzer.find_related_errors(
            error_record, recent_errors
        )
        
        # Store in database
        self.database.store_error(error_record)
        
        # Add to cache
        with self._lock:
            self.error_cache.append(error_record)
        
        # Log error
        self._log_error(error_record)
        
        return error_record
    
    def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """Mark an error as resolved with resolution notes."""
        error_record = self.database.get_error(error_id)
        if error_record:
            error_record.resolved = True
            error_record.resolution_notes = resolution_notes
            return self.database.store_error(error_record)
        return False
    
    def get_error_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive error analysis."""
        stats = self.database.get_error_statistics(hours_back)
        retry_stats = self.retry_manager.get_retry_stats()
        
        # Get recent critical errors
        critical_errors = self.database.get_errors_by_pattern(
            error_category=ErrorCategory.CRITICAL,
            hours_back=hours_back,
            limit=10
        )
        
        # Get unresolved high-severity errors
        high_severity_errors = self.database.get_errors_by_pattern(
            error_category=ErrorCategory.HIGH,
            hours_back=hours_back * 7,  # Look back a week for high severity
            limit=20
        )
        unresolved_high = [e for e in high_severity_errors if not e.resolved]
        
        return {
            "statistics": stats,
            "retry_statistics": retry_stats,
            "critical_errors": [e.to_dict() for e in critical_errors],
            "unresolved_high_severity": [e.to_dict() for e in unresolved_high],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_suggested_fixes(self, error_patterns: List[str]) -> List[str]:
        """Get suggested fixes for specific error patterns."""
        all_suggestions = set()
        
        for pattern in error_patterns:
            # Analyze pattern and get category
            category, _, suggestions = self.pattern_analyzer.analyze_error(pattern, "", "")
            all_suggestions.update(suggestions)
        
        return list(all_suggestions)
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level based on severity."""
        log_message = (
            f"Error {error_record.error_id}: {error_record.error_type} - "
            f"{error_record.error_message} "
            f"[{error_record.context.agent_id}/{error_record.context.tool_name}]"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _start_background_analysis(self):
        """Start background task for periodic error analysis."""
        def analyze_patterns():
            while True:
                try:
                    # Perform periodic analysis every hour
                    time.sleep(3600)
                    self._perform_pattern_analysis()
                except Exception as e:
                    logger.error(f"Background analysis error: {e}")
        
        analysis_thread = threading.Thread(target=analyze_patterns, daemon=True)
        analysis_thread.start()
    
    def _perform_pattern_analysis(self):
        """Perform periodic pattern analysis on recent errors."""
        try:
            # Get recent errors
            recent_errors = self.database.get_errors_by_pattern(hours_back=24, limit=100)
            
            # Analyze patterns and update related errors
            for error in recent_errors:
                if not error.related_errors:  # Only update if not already analyzed
                    related = self.pattern_analyzer.find_related_errors(error, recent_errors)
                    if related:
                        error.related_errors = related
                        self.database.store_error(error)
            
            logger.info(f"Completed pattern analysis for {len(recent_errors)} recent errors")
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")


# Global error handler instance
_error_handler = CentralizedErrorHandler()


def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic with exponential backoff to functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_config = config or _error_handler.retry_manager.default_config
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful attempt
                    _error_handler.retry_manager.record_retry_attempt(
                        func.__name__, attempt, True
                    )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not _error_handler.retry_manager.should_retry(e, attempt, retry_config):
                        break
                    
                    # Calculate delay
                    if attempt < retry_config.max_attempts:
                        delay = _error_handler.retry_manager.calculate_delay(attempt, retry_config)
                        if delay > 0:
                            time.sleep(delay)
                    
                    # Record retry attempt
                    _error_handler.retry_manager.record_retry_attempt(
                        func.__name__, attempt, False
                    )
            
            # All retries exhausted, handle the error
            context = ErrorContext(
                agent_id=kwargs.get('agent_id', 'unknown'),
                tool_name=kwargs.get('tool_name', func.__module__),
                function_name=func.__name__,
                parameters={k: str(v) for k, v in kwargs.items()},
                timestamp=datetime.now()
            )
            
            error_record = _error_handler.handle_error(last_exception, context)
            
            # Re-raise the exception with error ID
            raise type(last_exception)(
                f"{str(last_exception)} [Error ID: {error_record.error_id}]"
            ) from last_exception
        
        return wrapper
    return decorator


def with_async_retry(config: RetryConfig = None):
    """Async version of retry decorator."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_config = config or _error_handler.retry_manager.default_config
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record successful attempt
                    _error_handler.retry_manager.record_retry_attempt(
                        func.__name__, attempt, True
                    )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not _error_handler.retry_manager.should_retry(e, attempt, retry_config):
                        break
                    
                    # Calculate delay
                    if attempt < retry_config.max_attempts:
                        delay = _error_handler.retry_manager.calculate_delay(attempt, retry_config)
                        if delay > 0:
                            await asyncio.sleep(delay)
                    
                    # Record retry attempt
                    _error_handler.retry_manager.record_retry_attempt(
                        func.__name__, attempt, False
                    )
            
            # All retries exhausted, handle the error
            context = ErrorContext(
                agent_id=kwargs.get('agent_id', 'unknown'),
                tool_name=kwargs.get('tool_name', func.__module__),
                function_name=func.__name__,
                parameters={k: str(v) for k, v in kwargs.items()},
                timestamp=datetime.now()
            )
            
            error_record = _error_handler.handle_error(last_exception, context)
            
            # Re-raise the exception with error ID
            raise type(last_exception)(
                f"{str(last_exception)} [Error ID: {error_record.error_id}]"
            ) from last_exception
        
        return wrapper
    return decorator


# Input models for the error handling tools
class LogErrorInput(BaseModel):
    exception_type: str = Field(description="Type of exception")
    error_message: str = Field(description="Error message")
    agent_id: str = Field(description="Agent that encountered the error")
    tool_name: str = Field(description="Tool where error occurred")
    function_name: str = Field(description="Function where error occurred")
    parameters: Dict[str, Any] = Field(description="Function parameters")
    stack_trace: str = Field(default="", description="Stack trace")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    environment: str = Field(default="production", description="Environment")


class ResolveErrorInput(BaseModel):
    error_id: str = Field(description="Error ID to resolve")
    resolution_notes: str = Field(description="Resolution notes")


class GetErrorAnalysisInput(BaseModel):
    hours_back: int = Field(default=24, description="Hours to look back for analysis")
    include_suggestions: bool = Field(default=True, description="Include fix suggestions")


class GetErrorsByPatternInput(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="Filter by agent ID")
    tool_name: Optional[str] = Field(default=None, description="Filter by tool name")
    error_category: Optional[str] = Field(default=None, description="Filter by error category")
    hours_back: int = Field(default=24, description="Hours to look back")
    limit: int = Field(default=100, description="Maximum number of errors to return")


# Error handling tools
@tool(args_schema=LogErrorInput)
def log_error_centralized(
    exception_type: str,
    error_message: str,
    agent_id: str,
    tool_name: str,
    function_name: str,
    parameters: Dict[str, Any],
    stack_trace: str = "",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    environment: str = "production"
) -> str:
    """
    Log an error to the centralized error handling system with analysis and suggestions.
    
    Args:
        exception_type: Type of exception that occurred
        error_message: Error message
        agent_id: Agent that encountered the error
        tool_name: Tool where error occurred
        function_name: Function where error occurred
        parameters: Function parameters when error occurred
        stack_trace: Stack trace of the error
        session_id: Session ID if available
        user_id: User ID if available
        environment: Environment where error occurred
    
    Returns:
        JSON string with error record and analysis
    """
    try:
        # Create error context
        context = ErrorContext(
            agent_id=agent_id,
            tool_name=tool_name,
            function_name=function_name,
            parameters=parameters,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            environment=environment
        )
        
        # Create a mock exception for analysis
        class MockException(Exception):
            pass
        
        MockException.__name__ = exception_type
        mock_exception = MockException(error_message)
        
        # Handle the error
        error_record = _error_handler.handle_error(mock_exception, context)
        
        # Add stack trace if provided
        if stack_trace:
            error_record.stack_trace = stack_trace
            _error_handler.database.store_error(error_record)
        
        return json.dumps({
            "status": "success",
            "error_record": error_record.to_dict(),
            "message": f"Error logged with ID: {error_record.error_id}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to log error: {str(e)}"
        })


@tool(args_schema=ResolveErrorInput)
def resolve_error_centralized(error_id: str, resolution_notes: str) -> str:
    """
    Mark an error as resolved with resolution notes.
    
    Args:
        error_id: Error ID to resolve
        resolution_notes: Notes describing how the error was resolved
    
    Returns:
        JSON string with resolution status
    """
    try:
        success = _error_handler.resolve_error(error_id, resolution_notes)
        
        if success:
            return json.dumps({
                "status": "success",
                "message": f"Error {error_id} marked as resolved"
            })
        else:
            return json.dumps({
                "status": "failure",
                "message": f"Error {error_id} not found or could not be resolved"
            })
            
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to resolve error: {str(e)}"
        })


@tool(args_schema=GetErrorAnalysisInput)
def get_error_analysis(hours_back: int = 24, include_suggestions: bool = True) -> str:
    """
    Get comprehensive error analysis including patterns, statistics, and suggestions.
    
    Args:
        hours_back: Hours to look back for analysis
        include_suggestions: Include fix suggestions in the analysis
    
    Returns:
        JSON string with comprehensive error analysis
    """
    try:
        analysis = _error_handler.get_error_analysis(hours_back)
        
        if include_suggestions and analysis.get("statistics", {}).get("top_error_types"):
            # Get suggestions for top error types
            top_errors = analysis["statistics"]["top_error_types"]
            error_patterns = [error["error_type"] for error in top_errors[:5]]
            suggestions = _error_handler.get_suggested_fixes(error_patterns)
            analysis["suggested_fixes"] = suggestions
        
        return json.dumps({
            "status": "success",
            "analysis": analysis,
            "message": f"Error analysis for last {hours_back} hours"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get error analysis: {str(e)}"
        })


@tool(args_schema=GetErrorsByPatternInput)
def get_errors_by_pattern(
    agent_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    error_category: Optional[str] = None,
    hours_back: int = 24,
    limit: int = 100
) -> str:
    """
    Get errors matching specific patterns for detailed analysis.
    
    Args:
        agent_id: Filter by agent ID
        tool_name: Filter by tool name
        error_category: Filter by error category
        hours_back: Hours to look back
        limit: Maximum number of errors to return
    
    Returns:
        JSON string with matching errors
    """
    try:
        category_enum = None
        if error_category:
            try:
                category_enum = ErrorCategory(error_category.lower())
            except ValueError:
                return json.dumps({
                    "status": "failure",
                    "message": f"Invalid error category: {error_category}"
                })
        
        errors = _error_handler.database.get_errors_by_pattern(
            agent_id=agent_id,
            tool_name=tool_name,
            error_category=category_enum,
            hours_back=hours_back,
            limit=limit
        )
        
        return json.dumps({
            "status": "success",
            "errors": [error.to_dict() for error in errors],
            "count": len(errors),
            "message": f"Found {len(errors)} matching errors"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get errors by pattern: {str(e)}"
        })


@tool
def get_retry_statistics() -> str:
    """
    Get retry statistics for monitoring retry effectiveness.
    
    Returns:
        JSON string with retry statistics
    """
    try:
        stats = _error_handler.retry_manager.get_retry_stats()
        
        # Calculate success rates
        success_rates = {}
        for key, value in stats.items():
            if "_success_after_" in key:
                function_name = key.split("_success_after_")[0]
                attempt = key.split("_success_after_")[1]
                total_key = f"{function_name}_attempts"
                if total_key in stats:
                    success_rate = (value / stats[total_key]) * 100
                    success_rates[f"{function_name}_attempt_{attempt}_success_rate"] = success_rate
        
        return json.dumps({
            "status": "success",
            "retry_statistics": stats,
            "success_rates": success_rates,
            "message": "Retry statistics retrieved successfully"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get retry statistics: {str(e)}"
        })


@tool
def get_error_categories_info() -> str:
    """
    Get information about available error categories and their descriptions.
    
    Returns:
        JSON string with error categories information
    """
    try:
        categories_info = {
            "error_categories": {
                category.value: {
                    "name": category.value,
                    "description": _get_category_description(category),
                    "typical_fixes": _error_handler.pattern_analyzer.fix_suggestions.get(category, [])
                }
                for category in ErrorCategory
            },
            "severity_levels": {
                severity.value: _get_severity_description(severity)
                for severity in ErrorSeverity
            },
            "retry_strategies": {
                strategy.value: _get_strategy_description(strategy)
                for strategy in RetryStrategy
            }
        }
        
        return json.dumps({
            "status": "success",
            "categories_info": categories_info,
            "message": "Error categories information retrieved successfully"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get categories info: {str(e)}"
        })


def _get_category_description(category: ErrorCategory) -> str:
    """Get description for error category."""
    descriptions = {
        ErrorCategory.NETWORK: "Network connectivity and communication errors",
        ErrorCategory.AUTHENTICATION: "Authentication and authorization failures",
        ErrorCategory.PERMISSION: "Permission and access control errors",
        ErrorCategory.RESOURCE: "Resource exhaustion and allocation errors",
        ErrorCategory.VALIDATION: "Input validation and data format errors",
        ErrorCategory.TIMEOUT: "Operation timeout and deadline exceeded errors",
        ErrorCategory.CONFIGURATION: "Configuration and setup errors",
        ErrorCategory.DEPENDENCY: "Missing dependencies and import errors",
        ErrorCategory.LOGIC: "Business logic and algorithm errors",
        ErrorCategory.UNKNOWN: "Unclassified or unknown errors"
    }
    return descriptions.get(category, "Unknown category")


def _get_severity_description(severity: ErrorSeverity) -> str:
    """Get description for error severity."""
    descriptions = {
        ErrorSeverity.CRITICAL: "System-breaking errors requiring immediate attention",
        ErrorSeverity.HIGH: "Serious errors affecting functionality",
        ErrorSeverity.MEDIUM: "Moderate errors with workarounds available",
        ErrorSeverity.LOW: "Minor errors with minimal impact",
        ErrorSeverity.INFO: "Informational messages and warnings"
    }
    return descriptions.get(severity, "Unknown severity")


def _get_strategy_description(strategy: RetryStrategy) -> str:
    """Get description for retry strategy."""
    descriptions = {
        RetryStrategy.EXPONENTIAL_BACKOFF: "Exponentially increasing delays between retries",
        RetryStrategy.LINEAR_BACKOFF: "Linearly increasing delays between retries",
        RetryStrategy.FIXED_DELAY: "Fixed delay between retries",
        RetryStrategy.IMMEDIATE: "Immediate retry without delay",
        RetryStrategy.NO_RETRY: "No retry attempts"
    }
    return descriptions.get(strategy, "Unknown strategy")


# Export main components
__all__ = [
    "CentralizedErrorHandler",
    "RetryManager",
    "ErrorPatternAnalyzer",
    "with_retry",
    "with_async_retry",
    "log_error_centralized",
    "resolve_error_centralized",
    "get_error_analysis",
    "get_errors_by_pattern",
    "get_retry_statistics",
    "get_error_categories_info"
]