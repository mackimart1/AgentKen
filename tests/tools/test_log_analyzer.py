"""
Test suite for log_analyzer tool.
Tests comprehensive log analysis functionality including pattern recognition and anomaly detection.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, mock_open

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tools.log_analyzer import (
    analyze_log_file, 
    analyze_log_content, 
    find_log_files, 
    analyze_error_patterns,
    LogAnalyzer,
    LogEntry,
    LogLevel
)


class TestLogAnalyzer(unittest.TestCase):
    """Test cases for log analyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_log_content = """
2025-01-16 10:00:01 INFO agent_smith: Starting agent creation task
2025-01-16 10:00:02 DEBUG config: Loading configuration from file
2025-01-16 10:00:03 WARNING memory_manager: Memory usage at 85%
2025-01-16 10:00:04 ERROR tool_maker: Failed to create tool: FileNotFoundError
2025-01-16 10:00:05 INFO hermes: Task completed successfully
2025-01-16 10:00:06 ERROR tool_maker: Failed to create tool: FileNotFoundError
2025-01-16 10:00:07 CRITICAL system: Database connection lost
2025-01-16 10:00:08 INFO system: Attempting reconnection
2025-01-16 10:00:09 ERROR network: Connection timeout after 30 seconds
2025-01-16 10:00:10 INFO system: Database connection restored
        """.strip()

        self.error_log_content = """
2025-01-16 10:00:01 ERROR api: Request failed with status 500
2025-01-16 10:00:02 ERROR api: Request failed with status 404
2025-01-16 10:00:03 ERROR api: Request failed with status 500
2025-01-16 10:00:04 ERROR database: Connection timeout
2025-01-16 10:00:05 ERROR api: Request failed with status 500
2025-01-16 10:00:06 ERROR database: Connection timeout
        """.strip()

    def test_analyze_log_content_basic(self):
        """Test basic log content analysis."""
        result_str = analyze_log_content(self.sample_log_content)
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("analysis", result)
        
        analysis = result["analysis"]
        self.assertGreater(analysis["total_entries"], 0)
        self.assertGreater(analysis["info_count"], 0)
        self.assertGreater(analysis["error_count"], 0)
        self.assertGreater(analysis["warning_count"], 0)
        self.assertGreater(analysis["critical_count"], 0)

    def test_analyze_log_content_empty(self):
        """Test analysis with empty content."""
        result_str = analyze_log_content("")
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Empty log content", result["message"])

    def test_analyze_log_content_error_patterns(self):
        """Test error pattern detection."""
        result_str = analyze_log_content(self.sample_log_content)
        result = json.loads(result_str)
        
        analysis = result["analysis"]
        self.assertIn("error_patterns", analysis)
        self.assertIn("unique_errors", analysis)
        self.assertGreater(len(analysis["unique_errors"]), 0)

    def test_analyze_log_file_success(self):
        """Test successful log file analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(self.sample_log_content)
            temp_file = f.name

        try:
            result_str = analyze_log_file(temp_file)
            result = json.loads(result_str)
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["file_path"], temp_file)
            self.assertIn("analysis", result)
            self.assertGreater(result["lines_analyzed"], 0)
        finally:
            os.unlink(temp_file)

    def test_analyze_log_file_not_found(self):
        """Test analysis with non-existent file."""
        result_str = analyze_log_file("/nonexistent/file.log")
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "FileNotFoundError")

    def test_analyze_log_file_max_lines(self):
        """Test max lines limitation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            # Write more lines than max_lines
            for i in range(20):
                f.write(f"2025-01-16 10:00:{i:02d} INFO test: Log line {i}\n")
            temp_file = f.name

        try:
            result_str = analyze_log_file.invoke({"file_path": temp_file, "max_lines": 5})
            result = json.loads(result_str)
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["lines_analyzed"], 5)
        finally:
            os.unlink(temp_file)

    def test_find_log_files_success(self):
        """Test finding log files in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test log files
            log_files = ["test1.log", "test2.log", "other.txt"]
            for filename in log_files:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write("test content")

            result_str = find_log_files.invoke({"directory": temp_dir, "pattern": "*.log"})
            result = json.loads(result_str)
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["files_found"], 2)  # Only .log files
            self.assertIn("files", result)

    def test_find_log_files_directory_not_found(self):
        """Test finding files in non-existent directory."""
        result_str = find_log_files("/nonexistent/directory")
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Directory not found", result["message"])

    def test_analyze_error_patterns_basic(self):
        """Test error pattern analysis."""
        result_str = analyze_error_patterns(self.error_log_content)
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["total_errors"], 0)
        self.assertGreater(result["unique_patterns"], 0)
        self.assertIn("patterns", result)

    def test_analyze_error_patterns_min_occurrences(self):
        """Test error pattern analysis with minimum occurrences filter."""
        result_str = analyze_error_patterns.invoke({"log_content": self.error_log_content, "min_occurrences": 3})
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        # Should find the "Request failed with status N" pattern (occurs 3 times)
        patterns = result["patterns"]
        self.assertGreater(len(patterns), 0)
        
        # Check that patterns have at least 3 occurrences
        for pattern in patterns:
            self.assertGreaterEqual(pattern["count"], 3)

    def test_analyze_error_patterns_no_errors(self):
        """Test error pattern analysis with no errors."""
        info_only_content = """
2025-01-16 10:00:01 INFO test: Starting process
2025-01-16 10:00:02 INFO test: Process completed
        """.strip()
        
        result_str = analyze_error_patterns(info_only_content)
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(len(result["patterns"]), 0)

    def test_log_analyzer_class_basic(self):
        """Test LogAnalyzer class directly."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs(self.sample_log_content)
        
        self.assertGreater(result.total_entries, 0)
        self.assertGreater(result.error_count, 0)
        self.assertGreater(result.info_count, 0)
        self.assertIsInstance(result.unique_errors, list)
        self.assertIsInstance(result.error_patterns, dict)

    def test_log_entry_parsing(self):
        """Test log entry parsing functionality."""
        analyzer = LogAnalyzer()
        test_line = "2025-01-16 10:00:01 ERROR tool_maker: Failed to create tool"
        
        entry = analyzer.parse_log_line(test_line, 1)
        
        self.assertIsInstance(entry, LogEntry)
        self.assertEqual(entry.level, LogLevel.ERROR)
        self.assertIsNotNone(entry.timestamp)
        self.assertIn("Failed to create tool", entry.message)
        self.assertEqual(entry.line_number, 1)

    def test_log_entry_to_dict(self):
        """Test LogEntry to dictionary conversion."""
        analyzer = LogAnalyzer()
        test_line = "2025-01-16 10:00:01 INFO test: Test message"
        
        entry = analyzer.parse_log_line(test_line, 1)
        entry_dict = entry.to_dict()
        
        self.assertIsInstance(entry_dict, dict)
        self.assertIn("timestamp", entry_dict)
        self.assertIn("level", entry_dict)
        self.assertIn("message", entry_dict)
        self.assertIn("line_number", entry_dict)

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs(self.sample_log_content)
        
        self.assertIsInstance(result.performance_metrics, dict)
        metrics = result.performance_metrics
        self.assertIn("error_rate", metrics)
        self.assertIn("entries_per_minute", metrics)

    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Create content with high error rate
        high_error_content = """
2025-01-16 10:00:01 ERROR test: Error 1
2025-01-16 10:00:02 ERROR test: Error 2
2025-01-16 10:00:03 ERROR test: Error 3
2025-01-16 10:00:04 INFO test: Info message
        """.strip()
        
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs(high_error_content)
        
        self.assertIsInstance(result.anomalies, list)
        # Should detect high error rate
        anomaly_types = [a.get("type") for a in result.anomalies]
        self.assertIn("high_error_rate", anomaly_types)

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs(self.sample_log_content)
        
        self.assertIsInstance(result.recommendations, list)
        self.assertGreater(len(result.recommendations), 0)
        
        # Check that recommendations are strings
        for recommendation in result.recommendations:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 0)

    def test_time_range_analysis(self):
        """Test time range analysis."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs(self.sample_log_content)
        
        self.assertIsInstance(result.time_range, dict)
        time_range = result.time_range
        self.assertIn("start", time_range)
        self.assertIn("end", time_range)
        
        if time_range["start"] and time_range["end"]:
            # Verify start is before or equal to end
            start_time = datetime.fromisoformat(time_range["start"])
            end_time = datetime.fromisoformat(time_range["end"])
            self.assertLessEqual(start_time, end_time)

    def test_large_log_handling(self):
        """Test handling of large log content."""
        # Generate large log content
        large_content = []
        for i in range(1000):
            large_content.append(f"2025-01-16 10:{i//60:02d}:{i%60:02d} INFO test: Log entry {i}")
        
        large_log = "\n".join(large_content)
        
        result_str = analyze_log_content(large_log)
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        analysis = result["analysis"]
        self.assertEqual(analysis["total_entries"], 1000)

    def test_malformed_log_handling(self):
        """Test handling of malformed log entries."""
        malformed_content = """
This is not a proper log line
2025-01-16 10:00:01 INFO test: This is proper
Another malformed line without timestamp
ERROR: This has level but no timestamp
2025-01-16 10:00:02 DEBUG test: Another proper line
        """.strip()
        
        result_str = analyze_log_content(malformed_content)
        result = json.loads(result_str)
        
        self.assertEqual(result["status"], "success")
        # Should still process the content, even with malformed lines
        self.assertGreater(result["analysis"]["total_entries"], 0)


if __name__ == "__main__":
    unittest.main()