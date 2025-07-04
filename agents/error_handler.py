from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
import re
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the enhanced state for the agent
class AgentState(TypedDict):
    error_message: str
    error_type: str
    severity: str
    analysis: str
    suggested_fixes: List[str]
    attempted_fix_result: str
    context: Dict[str, str]
    timestamp: str
    confidence_score: float

# Error patterns and their classifications
ERROR_PATTERNS = {
    # File system errors
    r"(file not found|no such file or directory)": {
        "type": "FileSystemError",
        "severity": "Medium",
        "fixes": [
            "Verify the file path is correct",
            "Check if the file exists in the expected location",
            "Ensure proper file permissions",
            "Create the missing file if needed"
        ]
    },
    r"(permission denied|access denied)": {
        "type": "PermissionError", 
        "severity": "Medium",
        "fixes": [
            "Check file/directory permissions (chmod)",
            "Run with elevated privileges if necessary",
            "Verify user ownership (chown)",
            "Check parent directory permissions"
        ]
    },
    r"(disk full|no space left)": {
        "type": "DiskSpaceError",
        "severity": "High", 
        "fixes": [
            "Free up disk space by removing unnecessary files",
            "Move files to another partition",
            "Clean temporary files and logs",
            "Increase disk capacity"
        ]
    },
    
    # Network errors
    r"(connection refused|connection timeout)": {
        "type": "NetworkError",
        "severity": "Medium",
        "fixes": [
            "Check if the target service is running",
            "Verify network connectivity",
            "Check firewall settings",
            "Validate host/port configuration"
        ]
    },
    r"(dns resolution failed|name resolution)": {
        "type": "DNSError",
        "severity": "Medium",
        "fixes": [
            "Check DNS server configuration",
            "Verify hostname spelling",
            "Try using IP address instead",
            "Check /etc/hosts file"
        ]
    },
    
    # Programming errors
    r"(syntax error|syntaxerror)": {
        "type": "SyntaxError",
        "severity": "Low",
        "fixes": [
            "Review code syntax for typos",
            "Check bracket/parentheses matching",
            "Verify indentation (for Python)",
            "Use IDE syntax highlighting"
        ]
    },
    r"(name error|nameerror|undefined variable)": {
        "type": "NameError",
        "severity": "Low",
        "fixes": [
            "Check variable/function name spelling",
            "Ensure variable is defined before use",
            "Check import statements",
            "Verify scope of variable declaration"
        ]
    },
    r"(type error|typeerror)": {
        "type": "TypeError",
        "severity": "Medium",
        "fixes": [
            "Check data types being used",
            "Verify function arguments match expected types",
            "Add type conversion if needed",
            "Review API documentation for correct usage"
        ]
    },
    r"(import error|importerror|module not found)": {
        "type": "ImportError",
        "severity": "Medium",
        "fixes": [
            "Install missing package/module",
            "Check PYTHONPATH environment variable",
            "Verify module name spelling",
            "Use virtual environment if needed"
        ]
    },
    
    # System errors
    r"(command not found|command not recognized)": {
        "type": "CommandError",
        "severity": "Medium",
        "fixes": [
            "Install the required software/package",
            "Add command directory to PATH",
            "Use full path to command",
            "Check command spelling"
        ]
    },
    r"(out of memory|memory error)": {
        "type": "MemoryError",
        "severity": "High",
        "fixes": [
            "Optimize code to use less memory",
            "Process data in smaller chunks",
            "Close unused resources",
            "Increase available RAM"
        ]
    },
    r"(segmentation fault|segfault)": {
        "type": "SegmentationFault",
        "severity": "High",
        "fixes": [
            "Check for buffer overflows",
            "Verify pointer operations",
            "Use debugging tools (gdb, valgrind)",
            "Review memory allocation/deallocation"
        ]
    },
    
    # Database errors
    r"(connection to database failed|database connection error)": {
        "type": "DatabaseConnectionError",
        "severity": "High",
        "fixes": [
            "Check database server status",
            "Verify connection credentials",
            "Check network connectivity to database",
            "Review connection string/URL"
        ]
    },
    r"(sql syntax error|invalid sql)": {
        "type": "SQLError",
        "severity": "Medium",
        "fixes": [
            "Review SQL query syntax",
            "Check table/column names",
            "Verify SQL dialect compatibility",
            "Use SQL validation tools"
        ]
    }
}

def extract_error_context(error_message: str) -> Dict[str, str]:
    """Extract contextual information from error message"""
    context = {}
    
    # Extract file paths
    file_path_pattern = r'["\']?([/\\]?[\w\-./\\]+\.\w+)["\']?'
    file_matches = re.findall(file_path_pattern, error_message)
    if file_matches:
        context["file_path"] = file_matches[0]
    
    # Extract line numbers
    line_pattern = r'line (\d+)'
    line_matches = re.findall(line_pattern, error_message, re.IGNORECASE)
    if line_matches:
        context["line_number"] = line_matches[0]
    
    # Extract port numbers
    port_pattern = r'port (\d+)'
    port_matches = re.findall(port_pattern, error_message, re.IGNORECASE)
    if port_matches:
        context["port"] = port_matches[0]
    
    # Extract hostnames/IPs
    host_pattern = r'(?:host|server|address)[:=\s]+([a-zA-Z0-9.-]+)'
    host_matches = re.findall(host_pattern, error_message, re.IGNORECASE)
    if host_matches:
        context["host"] = host_matches[0]
    
    return context

def calculate_confidence_score(error_message: str, matched_patterns: List[str]) -> float:
    """Calculate confidence score based on pattern matches and message quality"""
    base_score = 0.5
    
    # Boost score for multiple pattern matches
    pattern_boost = min(len(matched_patterns) * 0.2, 0.4)
    
    # Boost score for detailed error messages
    detail_boost = min(len(error_message) / 200, 0.3)
    
    # Penalize for very short or generic messages
    if len(error_message) < 20:
        detail_boost = -0.2
    
    return min(base_score + pattern_boost + detail_boost, 1.0)

def analyze_error(state: AgentState) -> AgentState:
    """Enhanced error analysis with pattern matching and context extraction"""
    error_message = state.get("error_message", "")
    
    if not error_message.strip():
        return {
            "error_type": "InvalidInput",
            "severity": "Low",
            "analysis": "No error message provided",
            "suggested_fixes": ["Provide a valid error message for analysis"],
            "attempted_fix_result": "No fix attempted",
            "context": {},
            "timestamp": datetime.now().isoformat(),
            "confidence_score": 0.0
        }
    
    # Initialize analysis results
    error_type = "UnknownError"
    severity = "Low"
    suggested_fixes = ["Review logs for more details", "Contact system administrator"]
    matched_patterns = []
    
    # Pattern matching
    for pattern, info in ERROR_PATTERNS.items():
        if re.search(pattern, error_message, re.IGNORECASE):
            error_type = info["type"]
            severity = info["severity"]
            suggested_fixes = info["fixes"]
            matched_patterns.append(pattern)
            break  # Use first match for primary classification
    
    # Extract context
    context = extract_error_context(error_message)
    
    # Calculate confidence
    confidence_score = calculate_confidence_score(error_message, matched_patterns)
    
    # Generate detailed analysis
    analysis_parts = [
        f"Error Classification: {error_type}",
        f"Severity Level: {severity}",
        f"Pattern Matches: {len(matched_patterns)} found"
    ]
    
    if context:
        analysis_parts.append(f"Context Extracted: {', '.join(f'{k}={v}' for k, v in context.items())}")
    
    analysis = ". ".join(analysis_parts)
    
    logger.info(f"Analyzed error: {error_type} (confidence: {confidence_score:.2f})")
    
    return {
        "error_type": error_type,
        "severity": severity,
        "analysis": analysis,
        "suggested_fixes": suggested_fixes,
        "attempted_fix_result": "Analysis completed",
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "confidence_score": confidence_score
    }

def format_output(state: AgentState) -> AgentState:
    """Format the final output for better readability"""
    formatted_output = {
        "Error Analysis Report": {
            "Timestamp": state.get("timestamp", ""),
            "Error Type": state.get("error_type", ""),
            "Severity": state.get("severity", ""),
            "Confidence Score": f"{state.get('confidence_score', 0):.2f}",
            "Analysis": state.get("analysis", ""),
            "Context": state.get("context", {}),
            "Suggested Fixes": state.get("suggested_fixes", [])
        }
    }
    
    return {**state, "formatted_report": json.dumps(formatted_output, indent=2)}

# Build the enhanced workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze_error", analyze_error)
workflow.add_node("format_output", format_output)

# Define the flow
workflow.set_entry_point("analyze_error")
workflow.add_edge("analyze_error", "format_output")
workflow.add_edge("format_output", END)

# Compile the agent
error_handler_agent = workflow.compile()

def handle_error(error_message: str) -> Dict:
    """Convenience function to handle a single error message"""
    initial_state = {"error_message": error_message}
    result = error_handler_agent.invoke(initial_state)
    return result

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "FileNotFoundError: [Errno 2] No such file or directory: '/path/to/config.txt'",
        "PermissionError: [Errno 13] Permission denied: '/var/log/app.log'",
        "ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443): Connection refused",
        "SyntaxError: invalid syntax (script.py, line 42)",
        "ImportError: No module named 'requests'",
        "MemoryError: Unable to allocate 8.00 GiB for an array",
        "CommandError: 'docker' command not found",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "DatabaseError: connection to server failed: Connection refused",
        "Generic error message with no specific pattern"
    ]
    
    print("=" * 80)
    print("ERROR HANDLER AGENT - TEST RESULTS")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case}")
        
        result = handle_error(test_case)
        
        print(f"\nError Type: {result.get('error_type', 'N/A')}")
        print(f"Severity: {result.get('severity', 'N/A')}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"Analysis: {result.get('analysis', 'N/A')}")
        
        fixes = result.get('suggested_fixes', [])
        if fixes:
            print("Suggested Fixes:")
            for j, fix in enumerate(fixes, 1):
                print(f"  {j}. {fix}")
        
        context = result.get('context', {})
        if context:
            print(f"Context: {context}")
        
        print("-" * 50)