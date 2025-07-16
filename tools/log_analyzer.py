"""
Log Analyzer Tool: Comprehensive log analysis and monitoring for the AgentK system.

This tool provides advanced log analysis capabilities including:
- Log parsing and pattern recognition
- Error detection and classification
- Performance metrics extraction
- Trend analysis and anomaly detection
- Log aggregation and reporting
"""

from langchain_core.tools import tool
from typing import Dict, List, Any, Optional, Union
import re
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import os
import glob

# Setup logger
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogPattern(Enum):
    """Common log patterns for analysis."""
    ERROR_PATTERN = r"ERROR|CRITICAL|FATAL"
    WARNING_PATTERN = r"WARNING|WARN"
    TIMESTAMP_PATTERN = r"\d{4}-\d{2}-\d{2}[\s\t]\d{2}:\d{2}:\d{2}"
    IP_PATTERN = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    URL_PATTERN = r"https?://[^\s]+"
    EXCEPTION_PATTERN = r"Exception|Error|Traceback"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: Optional[datetime]
    level: Optional[LogLevel]
    message: str
    source: Optional[str]
    line_number: Optional[int]
    raw_line: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level.value if self.level else None,
            "message": self.message,
            "source": self.source,
            "line_number": self.line_number,
            "raw_line": self.raw_line
        }


@dataclass
class LogAnalysisResult:
    """Log analysis result container."""
    total_entries: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    critical_count: int
    unique_errors: List[str]
    error_patterns: Dict[str, int]
    time_range: Dict[str, Optional[str]]
    performance_metrics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "debug_count": self.debug_count,
            "critical_count": self.critical_count,
            "unique_errors": self.unique_errors,
            "error_patterns": self.error_patterns,
            "time_range": self.time_range,
            "performance_metrics": self.performance_metrics,
            "anomalies": self.anomalies,
            "recommendations": self.recommendations
        }


class LogAnalyzer:
    """Advanced log analyzer with pattern recognition and anomaly detection."""
    
    def __init__(self):
        self.entries: List[LogEntry] = []
        self.error_patterns = defaultdict(int)
        self.performance_metrics = {}
        
    def parse_log_line(self, line: str, line_number: int) -> LogEntry:
        """Parse a single log line into structured format."""
        # Extract timestamp
        timestamp_match = re.search(LogPattern.TIMESTAMP_PATTERN.value, line)
        timestamp = None
        if timestamp_match:
            try:
                timestamp_str = timestamp_match.group()
                # Handle different timestamp formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
                    try:
                        timestamp = datetime.strptime(timestamp_str.split('.')[0], fmt)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.debug(f"Could not parse timestamp: {e}")
        
        # Extract log level
        level = None
        for log_level in LogLevel:
            if log_level.value in line.upper():
                level = log_level
                break
        
        # Extract source/module information
        source = None
        source_patterns = [
            r"(\w+\.py)",  # Python files
            r"(\w+):",     # Module names followed by colon
            r"\[(\w+)\]"   # Bracketed module names
        ]
        for pattern in source_patterns:
            match = re.search(pattern, line)
            if match:
                source = match.group(1)
                break
        
        # Extract main message (remove timestamp, level, source)
        message = line
        if timestamp_match:
            message = message.replace(timestamp_match.group(), "").strip()
        if level:
            message = re.sub(rf"\b{level.value}\b", "", message, flags=re.IGNORECASE).strip()
        if source:
            message = message.replace(source, "").strip()
        
        # Clean up message
        message = re.sub(r"^[\s\-:]+", "", message).strip()
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            source=source,
            line_number=line_number,
            raw_line=line
        )
    
    def analyze_logs(self, log_content: str) -> LogAnalysisResult:
        """Perform comprehensive log analysis."""
        lines = log_content.strip().split('\n')
        self.entries = []
        
        # Parse all log entries
        for i, line in enumerate(lines, 1):
            if line.strip():
                entry = self.parse_log_line(line.strip(), i)
                self.entries.append(entry)
        
        # Count by log level
        level_counts = Counter(entry.level for entry in self.entries if entry.level)
        
        # Analyze error patterns
        error_entries = [e for e in self.entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        unique_errors = list(set(entry.message for entry in error_entries))
        
        # Pattern analysis
        error_patterns = {}
        for entry in error_entries:
            # Group similar errors
            simplified_error = re.sub(r'\d+', 'N', entry.message)  # Replace numbers
            simplified_error = re.sub(r"'[^']*'", "'X'", simplified_error)  # Replace quoted strings
            error_patterns[simplified_error] = error_patterns.get(simplified_error, 0) + 1
        
        # Time range analysis
        timestamps = [e.timestamp for e in self.entries if e.timestamp]
        time_range = {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None
        }
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Anomaly detection
        anomalies = self._detect_anomalies()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(level_counts, error_patterns)
        
        return LogAnalysisResult(
            total_entries=len(self.entries),
            error_count=level_counts.get(LogLevel.ERROR, 0),
            warning_count=level_counts.get(LogLevel.WARNING, 0),
            info_count=level_counts.get(LogLevel.INFO, 0),
            debug_count=level_counts.get(LogLevel.DEBUG, 0),
            critical_count=level_counts.get(LogLevel.CRITICAL, 0),
            unique_errors=unique_errors[:10],  # Top 10 unique errors
            error_patterns=dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            time_range=time_range,
            performance_metrics=performance_metrics,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        metrics = {
            "entries_per_minute": 0,
            "error_rate": 0,
            "most_active_source": None,
            "peak_activity_time": None
        }
        
        if not self.entries:
            return metrics
        
        # Calculate entries per minute
        timestamps = [e.timestamp for e in self.entries if e.timestamp]
        if len(timestamps) >= 2:
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60
            if time_span > 0:
                metrics["entries_per_minute"] = len(self.entries) / time_span
        
        # Calculate error rate
        error_count = sum(1 for e in self.entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        metrics["error_rate"] = error_count / len(self.entries) if self.entries else 0
        
        # Most active source
        source_counts = Counter(e.source for e in self.entries if e.source)
        if source_counts:
            metrics["most_active_source"] = source_counts.most_common(1)[0][0]
        
        # Peak activity time (hour of day)
        if timestamps:
            hour_counts = Counter(ts.hour for ts in timestamps)
            if hour_counts:
                metrics["peak_activity_time"] = f"{hour_counts.most_common(1)[0][0]:02d}:00"
        
        return metrics
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns."""
        anomalies = []
        
        # High error rate anomaly
        error_count = sum(1 for e in self.entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        if self.entries and error_count / len(self.entries) > 0.1:  # More than 10% errors
            anomalies.append({
                "type": "high_error_rate",
                "severity": "high",
                "description": f"High error rate detected: {error_count}/{len(self.entries)} ({error_count/len(self.entries)*100:.1f}%)",
                "recommendation": "Investigate recent changes or system issues"
            })
        
        # Repeated error patterns
        error_messages = [e.message for e in self.entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        message_counts = Counter(error_messages)
        for message, count in message_counts.items():
            if count >= 5:  # Same error repeated 5+ times
                anomalies.append({
                    "type": "repeated_error",
                    "severity": "medium",
                    "description": f"Repeated error detected {count} times: {message[:100]}...",
                    "recommendation": "Fix the underlying cause to prevent error repetition"
                })
        
        # Time gaps (no logs for extended periods)
        timestamps = sorted([e.timestamp for e in self.entries if e.timestamp])
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if gap > 3600:  # Gap longer than 1 hour
                anomalies.append({
                    "type": "logging_gap",
                    "severity": "low",
                    "description": f"Logging gap detected: {gap/3600:.1f} hours between {timestamps[i-1]} and {timestamps[i]}",
                    "recommendation": "Check if logging service was interrupted"
                })
        
        return anomalies
    
    def _generate_recommendations(self, level_counts: Counter, error_patterns: Dict[str, int]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Error-based recommendations
        total_entries = sum(level_counts.values())
        error_count = level_counts.get(LogLevel.ERROR, 0) + level_counts.get(LogLevel.CRITICAL, 0)
        
        if error_count > 0:
            error_rate = error_count / total_entries if total_entries > 0 else 0
            if error_rate > 0.05:  # More than 5% errors
                recommendations.append(f"High error rate ({error_rate*100:.1f}%) - investigate and fix critical issues")
            
            # Most common error pattern
            if error_patterns:
                most_common_error = list(error_patterns.keys())[0]
                recommendations.append(f"Most frequent error pattern: '{most_common_error}' - prioritize fixing this issue")
        
        # Warning-based recommendations
        warning_count = level_counts.get(LogLevel.WARNING, 0)
        if warning_count > error_count * 2:  # Many more warnings than errors
            recommendations.append("High warning count detected - review and address warnings to prevent future errors")
        
        # General recommendations
        if total_entries < 10:
            recommendations.append("Low log volume - consider increasing logging verbosity for better monitoring")
        
        if not any(level_counts.get(level, 0) > 0 for level in [LogLevel.DEBUG, LogLevel.INFO]):
            recommendations.append("No debug/info logs found - consider enabling more detailed logging for troubleshooting")
        
        return recommendations


# Global analyzer instance
log_analyzer = LogAnalyzer()


@tool
def analyze_log_file(file_path: str, max_lines: int = 1000) -> str:
    """
    Analyze a log file and return comprehensive analysis results.
    
    Args:
        file_path (str): Path to the log file to analyze
        max_lines (int): Maximum number of lines to analyze (default: 1000)
        
    Returns:
        str: JSON string containing detailed log analysis results
        
    Raises:
        FileNotFoundError: If the log file doesn't exist
        PermissionError: If unable to read the log file
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        # Read log file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Limit lines if specified
        if max_lines > 0:
            lines = lines[-max_lines:]  # Get last N lines
        
        log_content = ''.join(lines)
        
        # Perform analysis
        result = log_analyzer.analyze_logs(log_content)
        
        # Return as JSON string
        return json.dumps({
            "status": "success",
            "file_path": file_path,
            "lines_analyzed": len(lines),
            "analysis": result.to_dict()
        }, indent=2)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return json.dumps({
            "status": "error",
            "error_type": "FileNotFoundError",
            "message": str(e)
        })
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return json.dumps({
            "status": "error",
            "error_type": "PermissionError", 
            "message": str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error analyzing log file: {e}")
        return json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        })


@tool
def analyze_log_content(log_content: str) -> str:
    """
    Analyze log content directly and return comprehensive analysis results.
    
    Args:
        log_content (str): Raw log content to analyze
        
    Returns:
        str: JSON string containing detailed log analysis results
    """
    try:
        if not log_content or not log_content.strip():
            return json.dumps({
                "status": "error",
                "error_type": "ValueError",
                "message": "Empty log content provided"
            })
        
        # Perform analysis
        result = log_analyzer.analyze_logs(log_content)
        
        # Return as JSON string
        return json.dumps({
            "status": "success",
            "content_length": len(log_content),
            "analysis": result.to_dict()
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error analyzing log content: {e}")
        return json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        })


@tool
def find_log_files(directory: str, pattern: str = "*.log") -> str:
    """
    Find log files in a directory matching a pattern.
    
    Args:
        directory (str): Directory to search for log files
        pattern (str): File pattern to match (default: "*.log")
        
    Returns:
        str: JSON string containing list of found log files
    """
    try:
        if not os.path.exists(directory):
            return json.dumps({
                "status": "error",
                "error_type": "DirectoryNotFoundError",
                "message": f"Directory not found: {directory}"
            })
        
        # Find matching files
        search_pattern = os.path.join(directory, pattern)
        log_files = glob.glob(search_pattern)
        
        # Get file info
        file_info = []
        for file_path in log_files:
            try:
                stat = os.stat(file_path)
                file_info.append({
                    "path": file_path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "readable": os.access(file_path, os.R_OK)
                })
            except Exception as e:
                logger.warning(f"Could not get info for {file_path}: {e}")
        
        return json.dumps({
            "status": "success",
            "directory": directory,
            "pattern": pattern,
            "files_found": len(file_info),
            "files": file_info
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error finding log files: {e}")
        return json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        })


@tool
def analyze_error_patterns(log_content: str, min_occurrences: int = 2) -> str:
    """
    Analyze error patterns in log content and identify recurring issues.
    
    Args:
        log_content (str): Raw log content to analyze
        min_occurrences (int): Minimum occurrences to consider a pattern (default: 2)
        
    Returns:
        str: JSON string containing error pattern analysis
    """
    try:
        if not log_content or not log_content.strip():
            return json.dumps({
                "status": "error",
                "message": "Empty log content provided"
            })
        
        # Parse log entries
        lines = log_content.strip().split('\n')
        entries = []
        for i, line in enumerate(lines, 1):
            if line.strip():
                entry = log_analyzer.parse_log_line(line.strip(), i)
                entries.append(entry)
        
        # Extract error entries
        error_entries = [e for e in entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        
        if not error_entries:
            return json.dumps({
                "status": "success",
                "message": "No errors found in log content",
                "error_count": 0,
                "patterns": []
            })
        
        # Analyze patterns
        patterns = defaultdict(list)
        for entry in error_entries:
            # Normalize error message for pattern matching
            normalized = re.sub(r'\d+', 'N', entry.message)  # Replace numbers
            normalized = re.sub(r"'[^']*'", "'X'", normalized)  # Replace quoted strings
            normalized = re.sub(r'"[^"]*"', '"X"', normalized)  # Replace quoted strings
            normalized = re.sub(r'\b[A-Fa-f0-9]{8,}\b', 'HASH', normalized)  # Replace hashes
            
            patterns[normalized].append({
                "line_number": entry.line_number,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "original_message": entry.message,
                "source": entry.source
            })
        
        # Filter by minimum occurrences
        significant_patterns = {
            pattern: occurrences 
            for pattern, occurrences in patterns.items() 
            if len(occurrences) >= min_occurrences
        }
        
        # Sort by frequency
        sorted_patterns = sorted(
            significant_patterns.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Format results
        pattern_results = []
        for pattern, occurrences in sorted_patterns:
            pattern_results.append({
                "pattern": pattern,
                "count": len(occurrences),
                "first_occurrence": min(o.get("timestamp") for o in occurrences if o.get("timestamp")) or "unknown",
                "last_occurrence": max(o.get("timestamp") for o in occurrences if o.get("timestamp")) or "unknown",
                "sources": list(set(o.get("source") for o in occurrences if o.get("source"))),
                "examples": occurrences[:3]  # First 3 examples
            })
        
        return json.dumps({
            "status": "success",
            "total_errors": len(error_entries),
            "unique_patterns": len(patterns),
            "significant_patterns": len(significant_patterns),
            "patterns": pattern_results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error analyzing error patterns: {e}")
        return json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        })


# Export all tools
__all__ = [
    "analyze_log_file",
    "analyze_log_content", 
    "find_log_files",
    "analyze_error_patterns",
    "LogAnalyzer",
    "LogEntry",
    "LogAnalysisResult"
]