# Error Handling & Recovery System Documentation

## Overview

The Error Handling & Recovery System provides enterprise-grade error management capabilities for AgentK with two key improvements:

1. **Centralized Error-Handling Agent** - Logs exceptions, analyzes failure patterns, and suggests actionable fixes
2. **Retry Logic with Exponential Backoff** - Handles transient failures gracefully with intelligent recovery strategies

## Key Features

### 🎯 Centralized Error-Handling Agent

A specialized agent that provides comprehensive error management across the entire system.

**Core Capabilities:**
- **Error Classification**: Automatic categorization by type, severity, and impact
- **Pattern Analysis**: ML-powered pattern recognition and clustering
- **Fix Suggestions**: Context-aware actionable recommendations
- **Trend Monitoring**: Long-term error trend analysis and reporting
- **Resolution Tracking**: Complete lifecycle management of errors

**Error Categories:**
- **Network**: Connection timeouts, DNS failures, network unreachable
- **Authentication**: Invalid credentials, expired tokens, unauthorized access
- **Permission**: Access denied, insufficient privileges, forbidden operations
- **Resource**: Memory exhaustion, disk full, rate limits exceeded
- **Validation**: Invalid input format, schema errors, data validation failures
- **Timeout**: Operation timeouts, deadline exceeded, slow responses
- **Configuration**: Missing config files, invalid settings, environment issues
- **Dependency**: Missing packages, import errors, version conflicts
- **Logic**: Business logic errors, algorithm failures, edge cases
- **Unknown**: Unclassified errors requiring investigation

### 🔄 Retry Logic with Exponential Backoff

Intelligent retry mechanisms that handle transient failures automatically.

**Retry Strategies:**
- **Exponential Backoff**: Exponentially increasing delays (1s, 2s, 4s, 8s...)
- **Linear Backoff**: Linearly increasing delays (1s, 2s, 3s, 4s...)
- **Fixed Delay**: Consistent delay between retries
- **Immediate**: No delay between retries
- **No Retry**: Fail immediately without retry attempts

**Smart Exception Handling:**
- **Retryable Exceptions**: ConnectionError, TimeoutError, OSError
- **Non-Retryable Exceptions**: ValueError, TypeError, PermissionError
- **Custom Configuration**: Configurable exception lists per use case
- **Jitter Support**: Random variation to prevent thundering herd

## Architecture

### Core Components

#### CentralizedErrorHandler
Main error handling orchestrator that coordinates all error management activities.

```python
from tools.error_handling_system import CentralizedErrorHandler

error_handler = CentralizedErrorHandler()
error_record = error_handler.handle_error(exception, context)
```

#### ErrorPatternAnalyzer
Analyzes error patterns and provides intelligent categorization and fix suggestions.

```python
from tools.error_handling_system import ErrorPatternAnalyzer

analyzer = ErrorPatternAnalyzer()
category, severity, suggestions = analyzer.analyze_error(error_message, error_type, stack_trace)
```

#### RetryManager
Manages retry logic with various strategies and performance monitoring.

```python
from tools.error_handling_system import RetryManager, RetryConfig

retry_manager = RetryManager()
should_retry = retry_manager.should_retry(exception, attempt, config)
delay = retry_manager.calculate_delay(attempt, config)
```

#### ErrorDatabase
SQLite-based storage for error records with comprehensive querying capabilities.

```python
from tools.error_handling_system import ErrorDatabase

db = ErrorDatabase()
db.store_error(error_record)
errors = db.get_errors_by_pattern(agent_id="web_researcher", hours_back=24)
```

### Data Models

#### ErrorRecord
Comprehensive error record with full context and analysis.

```python
@dataclass
class ErrorRecord:
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
```

#### ErrorContext
Context information for comprehensive error tracking.

```python
@dataclass
class ErrorContext:
    agent_id: str
    tool_name: str
    function_name: str
    parameters: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: str = "production"
```

#### RetryConfig
Configuration for retry behavior and strategies.

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: List[type] = field(default_factory=list)
    non_retryable_exceptions: List[type] = field(default_factory=list)
```

## Error Handling Tools

### Core Error Management

#### log_error_centralized
Log an error to the centralized system with automatic analysis.

```python
log_error_centralized(
    exception_type="ConnectionError",
    error_message="Connection timeout while connecting to database",
    agent_id="database_agent",
    tool_name="database_connector",
    function_name="connect_to_db",
    parameters={"host": "db.example.com", "port": 5432},
    stack_trace="Traceback...",
    session_id="session_123",
    environment="production"
)
```

**Returns:**
```json
{
  "status": "success",
  "error_record": {
    "error_id": "uuid-123",
    "error_type": "ConnectionError",
    "error_category": "timeout",
    "severity": "medium",
    "suggested_fixes": [
      "Increase timeout values in configuration",
      "Check network connectivity and latency",
      "Implement retry logic with exponential backoff"
    ],
    "related_errors": ["uuid-456", "uuid-789"]
  }
}
```

#### resolve_error_centralized
Mark an error as resolved with resolution notes.

```python
resolve_error_centralized(
    error_id="uuid-123",
    resolution_notes="Fixed by updating connection timeout configuration and implementing connection pooling"
)
```

#### get_error_analysis
Get comprehensive error analysis with statistics and trends.

```python
get_error_analysis(
    hours_back=24,
    include_suggestions=True
)
```

**Returns:**
```json
{
  "status": "success",
  "analysis": {
    "statistics": {
      "total_errors": 45,
      "resolved_errors": 38,
      "resolution_rate": 84.4,
      "category_distribution": {
        "network": 15,
        "authentication": 8,
        "validation": 12
      },
      "top_error_types": [
        {"error_type": "ConnectionError", "count": 15},
        {"error_type": "ValidationError", "count": 12}
      ]
    },
    "suggested_fixes": [
      "Increase timeout values in configuration",
      "Implement connection pooling",
      "Add input validation"
    ]
  }
}
```

### Pattern Analysis

#### get_errors_by_pattern
Query errors by specific patterns for detailed analysis.

```python
get_errors_by_pattern(
    agent_id="web_researcher",
    tool_name="search_tool",
    error_category="network",
    hours_back=24,
    limit=50
)
```

#### get_retry_statistics
Monitor retry effectiveness and performance.

```python
get_retry_statistics()
```

**Returns:**
```json
{
  "status": "success",
  "retry_statistics": {
    "network_call_attempts": 25,
    "network_call_success_after_2": 15,
    "database_operation_attempts": 12
  },
  "success_rates": {
    "network_call_attempt_2_success_rate": 60.0,
    "database_operation_attempt_1_success_rate": 75.0
  }
}
```

#### get_error_categories_info
Get information about error categories and retry strategies.

```python
get_error_categories_info()
```

## Retry Decorators

### @with_retry
Decorator for synchronous functions with retry logic.

```python
from tools.error_handling_system import with_retry, RetryConfig, RetryStrategy

@with_retry(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
))
def flaky_network_call(url: str):
    response = requests.get(url, timeout=30)
    return response.json()

# Usage
try:
    data = flaky_network_call("https://api.example.com/data")
except Exception as e:
    # Error automatically logged with retry attempts
    print(f"Failed after retries: {e}")
```

### @with_async_retry
Decorator for asynchronous functions with retry logic.

```python
from tools.error_handling_system import with_async_retry

@with_async_retry(RetryConfig(
    max_attempts=4,
    base_delay=0.5,
    strategy=RetryStrategy.LINEAR_BACKOFF
))
async def async_database_operation():
    async with database.transaction():
        return await database.execute("SELECT * FROM users")

# Usage
try:
    result = await async_database_operation()
except Exception as e:
    print(f"Database operation failed: {e}")
```

### Custom Retry Configuration

```python
# Network operations with exponential backoff
network_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=True,
    retryable_exceptions=[ConnectionError, TimeoutError, requests.RequestException],
    non_retryable_exceptions=[ValueError, TypeError, PermissionError]
)

# Database operations with linear backoff
database_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    retryable_exceptions=[psycopg2.OperationalError, sqlite3.OperationalError]
)

# Validation operations with no retry
validation_retry_config = RetryConfig(
    max_attempts=1,
    strategy=RetryStrategy.NO_RETRY
)
```

## Error Handler Agent

### Specialized Error Management Agent

The `error_handler` agent provides centralized error management capabilities.

```python
from agents.error_handler import error_handler

# Analyze recent errors and get recommendations
result = error_handler("Analyze all errors from the last 24 hours and provide actionable recommendations")

# Investigate specific error patterns
result = error_handler("Investigate all network-related errors and suggest infrastructure improvements")

# Monitor system health
result = error_handler("Provide a comprehensive system health report with error trends")
```

**Agent Capabilities:**
- **Error Analysis**: Comprehensive analysis of error patterns and trends
- **Pattern Recognition**: Identify recurring issues and systemic problems
- **Fix Recommendations**: Provide specific, actionable solutions
- **System Health Monitoring**: Track overall system reliability
- **Resolution Coordination**: Coordinate error resolution across agents

## Usage Examples

### Basic Error Logging

```python
from tools.error_handling_system import log_error_centralized

try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    # Log the error with full context
    log_result = log_error_centralized(
        exception_type=type(e).__name__,
        error_message=str(e),
        agent_id="my_agent",
        tool_name="my_tool",
        function_name="risky_operation",
        parameters={"param1": "value1"},
        stack_trace=traceback.format_exc()
    )
    
    error_data = json.loads(log_result)
    if error_data["status"] == "success":
        error_id = error_data["error_record"]["error_id"]
        suggestions = error_data["error_record"]["suggested_fixes"]
        print(f"Error logged: {error_id}")
        print(f"Suggestions: {suggestions}")
```

### Retry with Custom Configuration

```python
from tools.error_handling_system import with_retry, RetryConfig, RetryStrategy

# Custom retry configuration for API calls
api_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=True
)

@with_retry(api_retry_config)
def call_external_api(endpoint: str, data: dict):
    response = requests.post(f"https://api.example.com/{endpoint}", json=data)
    response.raise_for_status()
    return response.json()

# Usage with automatic retry and error logging
try:
    result = call_external_api("users", {"name": "John", "email": "john@example.com"})
    print(f"API call successful: {result}")
except Exception as e:
    print(f"API call failed after retries: {e}")
    # Error is automatically logged with retry statistics
```

### Error Pattern Analysis

```python
from tools.error_handling_system import get_errors_by_pattern, get_error_analysis

# Analyze network errors for a specific agent
network_errors = get_errors_by_pattern(
    agent_id="web_researcher",
    error_category="network",
    hours_back=168  # Last week
)

network_data = json.loads(network_errors)
if network_data["status"] == "success":
    errors = network_data["errors"]
    print(f"Found {len(errors)} network errors")
    
    for error in errors[:5]:  # Show first 5
        print(f"- {error['error_type']}: {error['error_message']}")
        print(f"  Suggestions: {error['suggested_fixes'][:2]}")

# Get comprehensive system analysis
analysis = get_error_analysis(hours_back=24, include_suggestions=True)
analysis_data = json.loads(analysis)

if analysis_data["status"] == "success":
    stats = analysis_data["analysis"]["statistics"]
    print(f"Total errors: {stats['total_errors']}")
    print(f"Resolution rate: {stats['resolution_rate']:.1f}%")
    
    if "suggested_fixes" in analysis_data["analysis"]:
        print("Top system-wide recommendations:")
        for suggestion in analysis_data["analysis"]["suggested_fixes"][:3]:
            print(f"- {suggestion}")
```

### Error Resolution Tracking

```python
from tools.error_handling_system import resolve_error_centralized

# Mark an error as resolved
resolution_result = resolve_error_centralized(
    error_id="uuid-123",
    resolution_notes="Fixed by updating database connection pool settings and implementing circuit breaker pattern"
)

resolution_data = json.loads(resolution_result)
if resolution_data["status"] == "success":
    print("Error marked as resolved")
else:
    print(f"Failed to resolve error: {resolution_data['message']}")
```

## Integration with Existing System

### Automatic Error Logging

The system can be integrated with existing agents and tools to automatically log errors:

```python
# In existing agent code
try:
    result = some_operation()
except Exception as e:
    # Automatic error logging
    context = ErrorContext(
        agent_id="my_agent",
        tool_name="my_tool",
        function_name="some_operation",
        parameters={"param": "value"},
        timestamp=datetime.now()
    )
    
    error_record = error_handler.handle_error(e, context)
    # Error is automatically categorized and suggestions are generated
    
    # Re-raise with error ID for tracking
    raise type(e)(f"{str(e)} [Error ID: {error_record.error_id}]") from e
```

### Enhanced Tool Development

Tools can be enhanced with automatic retry and error handling:

```python
from langchain_core.tools import tool
from tools.error_handling_system import with_retry, RetryConfig

@tool
@with_retry(RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF))
def enhanced_web_search(query: str) -> str:
    """Enhanced web search with automatic retry and error logging."""
    try:
        # Perform web search
        results = search_engine.search(query)
        return json.dumps(results)
    except Exception as e:
        # Error is automatically logged and retried
        raise
```

## Configuration

### Error Categories Configuration

```python
# Custom error pattern rules
custom_patterns = {
    "custom_timeout": {
        "patterns": [r"custom.*timeout", r"operation.*expired"],
        "category": ErrorCategory.TIMEOUT,
        "severity": ErrorSeverity.HIGH,
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF
    }
}

# Add to pattern analyzer
analyzer = ErrorPatternAnalyzer()
analyzer.pattern_rules.update(custom_patterns)
```

### Retry Strategy Configuration

```python
# Global retry configuration
default_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=True
)

# Per-operation configurations
retry_configs = {
    "network_operations": RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    ),
    "database_operations": RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        strategy=RetryStrategy.LINEAR_BACKOFF
    ),
    "validation_operations": RetryConfig(
        max_attempts=1,
        strategy=RetryStrategy.NO_RETRY
    )
}
```

### Database Configuration

```python
# Custom database path
error_db = ErrorDatabase("custom_error_handling.db")

# Database optimization settings
error_db.execute("""
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = 10000;
""")
```

## Monitoring and Metrics

### Key Performance Indicators

**Error Metrics:**
- Total error count per time period
- Error rate by category and severity
- Resolution rate and time to resolution
- Recurring error patterns and frequency

**Retry Metrics:**
- Retry success rate by strategy
- Average retry attempts per operation
- Time spent in retry operations
- Resource usage during retries

**System Health Metrics:**
- Overall system reliability score
- Agent-specific error rates
- Tool-specific failure patterns
- Performance impact of error handling

### Monitoring Dashboard Data

```python
from tools.error_handling_system import get_error_analysis, get_retry_statistics

# Get comprehensive metrics
error_metrics = get_error_analysis(hours_back=24)
retry_metrics = get_retry_statistics()

# Parse metrics for dashboard
error_data = json.loads(error_metrics)
retry_data = json.loads(retry_metrics)

dashboard_data = {
    "error_rate": error_data["analysis"]["statistics"]["total_errors"],
    "resolution_rate": error_data["analysis"]["statistics"]["resolution_rate"],
    "category_distribution": error_data["analysis"]["statistics"]["category_distribution"],
    "retry_success_rates": retry_data["success_rates"],
    "top_error_types": error_data["analysis"]["statistics"]["top_error_types"]
}
```

## Best Practices

### Error Handling Guidelines

1. **Always Log with Context**: Include full context information for better analysis
2. **Use Appropriate Retry Strategies**: Match retry strategy to error type and operation
3. **Monitor and Analyze Patterns**: Regularly review error patterns and trends
4. **Implement Circuit Breakers**: Use circuit breaker pattern for external dependencies
5. **Track Resolution Effectiveness**: Monitor which fixes actually work

### Retry Logic Best Practices

1. **Choose Appropriate Strategies**: Use exponential backoff for network, linear for resources
2. **Set Reasonable Limits**: Don't retry indefinitely, set appropriate max attempts
3. **Use Jitter**: Add randomness to prevent thundering herd problems
4. **Monitor Performance**: Track retry effectiveness and resource usage
5. **Handle Non-Retryable Errors**: Fail fast for validation and logic errors

### Performance Optimization

1. **Database Indexing**: Ensure proper indexing on error database tables
2. **Batch Operations**: Batch error logging operations when possible
3. **Async Processing**: Use async processing for non-critical error analysis
4. **Cache Patterns**: Cache frequently accessed error patterns and suggestions
5. **Resource Limits**: Set appropriate resource limits for retry operations

## Troubleshooting

### Common Issues

#### High Error Rates
- **Symptom**: Sudden increase in error logging
- **Diagnosis**: Check system resources, network connectivity, external dependencies
- **Solution**: Scale resources, implement circuit breakers, add health checks

#### Retry Storms
- **Symptom**: Excessive retry attempts causing system overload
- **Diagnosis**: Check retry configurations and failure patterns
- **Solution**: Adjust retry limits, add jitter, implement backpressure

#### Database Performance
- **Symptom**: Slow error logging and querying
- **Diagnosis**: Check database size, indexing, and query patterns
- **Solution**: Optimize indexes, archive old data, tune database settings

#### Pattern Recognition Issues
- **Symptom**: Incorrect error categorization or poor suggestions
- **Diagnosis**: Review pattern rules and training data
- **Solution**: Update pattern rules, improve training data, tune algorithms

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Error handling system will provide detailed logs
from tools.error_handling_system import CentralizedErrorHandler

error_handler = CentralizedErrorHandler()
# Detailed logs will show pattern matching, categorization, and suggestion generation
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**: Advanced pattern recognition using ML models
2. **Predictive Error Detection**: Predict errors before they occur based on patterns
3. **Auto-Resolution**: Automatic resolution of common, well-understood errors
4. **Integration APIs**: REST APIs for external monitoring and alerting systems
5. **Advanced Analytics**: Deeper insights into error patterns and system health

### Integration Opportunities

1. **Monitoring Systems**: Integration with Prometheus, Grafana, ELK stack
2. **Alerting Platforms**: Integration with PagerDuty, Slack, email notifications
3. **CI/CD Pipelines**: Integration with deployment and testing pipelines
4. **Cloud Platforms**: Integration with AWS CloudWatch, Azure Monitor, GCP Logging
5. **APM Tools**: Integration with New Relic, Datadog, AppDynamics

## Conclusion

The Error Handling & Recovery System provides enterprise-grade error management capabilities that significantly improve system reliability and operational efficiency. With centralized error logging, intelligent pattern analysis, and robust retry mechanisms, the system transforms error chaos into organized, actionable intelligence.

**Key Benefits:**
- **Improved Reliability**: Automatic retry and recovery for transient failures
- **Better Visibility**: Comprehensive error tracking and analysis
- **Faster Resolution**: Actionable fix suggestions and pattern recognition
- **Proactive Prevention**: Trend analysis and predictive insights
- **Operational Efficiency**: Reduced manual error investigation and resolution

The system is designed to be production-ready with comprehensive monitoring, performance optimization, and enterprise-grade features that support complex, high-availability environments.