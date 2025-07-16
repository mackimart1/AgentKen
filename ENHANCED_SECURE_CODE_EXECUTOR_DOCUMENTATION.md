# Enhanced Secure Code Executor Documentation

## Overview

Enhanced Secure Code Executor is an enterprise-grade code execution system with three key improvements over the original Secure Code Executor:

1. **Sandboxing** - Docker containerization for secure system-level isolation
2. **Multi-Language Support** - Support for 9+ programming languages
3. **Resource Limits** - Comprehensive execution constraints and monitoring

## Key Features

### 🐳 Sandboxing

Enhanced Secure Code Executor provides complete system-level isolation through Docker containerization.

**Capabilities:**
- **Docker Containers**: Complete system-level isolation for each execution
- **Network Isolation**: Configurable network access with default denial
- **File System Protection**: Read-only or restricted file system access
- **User Isolation**: Code runs as unprivileged user (nobody)
- **Resource Containers**: Containerized CPU, memory, and process limits

**Security Benefits:**
- Complete isolation from host system
- Prevention of system compromise
- Protection against malicious code
- Secure multi-tenant execution
- Automatic cleanup after execution

### 🌐 Multi-Language Support

Comprehensive support for multiple programming languages with language-specific configurations.

**Supported Languages:**
- **Python**: Full Python 3.11 support with standard libraries
- **JavaScript**: Node.js 18 runtime with npm ecosystem
- **Bash**: Shell scripting with Alpine Linux environment
- **Ruby**: Ruby 3.2 interpreter with gem support
- **Go**: Go 1.21 compiler and runtime
- **Rust**: Rust 1.75 compiler with Cargo
- **Java**: OpenJDK 17 with compilation and execution
- **C++**: GCC compiler with C++17 support
- **C**: GCC compiler with C11 support

**Language Features:**
- Language-specific Docker images
- Optimized execution environments
- Security restrictions per language
- Native and compiled language support
- Package manager integration

### ⚡ Resource Limits

Comprehensive resource monitoring and enforcement to prevent abuse and ensure system stability.

**Resource Controls:**
- **Execution Timeout**: Configurable time limits (default: 30s)
- **Memory Limits**: RAM usage constraints (default: 512MB)
- **CPU Limits**: CPU usage percentage caps (default: 80%)
- **Process Limits**: Maximum subprocess count (default: 5)
- **Output Limits**: Stdout/stderr size limits (default: 1MB)
- **File Size Limits**: Maximum file creation size (default: 10MB)

**Monitoring Features:**
- Real-time resource usage tracking
- Automatic limit enforcement
- Performance metrics collection
- Resource violation detection
- Graceful timeout handling

## Architecture

### Enhanced Components

#### EnhancedSecureCodeExecutor
Core class providing secure, multi-language code execution.

```python
executor = EnhancedSecureCodeExecutor()
result = executor.execute_code(
    code="print('Hello World!')",
    language="python",
    environment="docker",
    resource_limits={"max_execution_time": 30}
)
```

#### DockerSandbox
Docker-based sandboxing for maximum security isolation.

```python
sandbox = DockerSandbox()
result = sandbox.execute_in_container(code, language, limits)
```

#### NativeSandbox
Native execution with resource monitoring for development environments.

```python
sandbox = NativeSandbox()
result = sandbox.execute_native(code, language, limits)
```

#### SecurityValidator
Pre-execution security validation and risk assessment.

```python
validator = SecurityValidator()
is_safe, violations = validator.validate_code(code, language, limits)
```

### Execution Environments

#### Docker Environment
Maximum security with complete system isolation.
- **Use Case**: Production environments, untrusted code
- **Security**: Complete isolation, no host access
- **Performance**: Slight overhead for container creation
- **Requirements**: Docker installation and access

#### Native Environment
Direct execution with resource monitoring.
- **Use Case**: Development environments, trusted code
- **Security**: Process-level isolation with monitoring
- **Performance**: Fastest execution times
- **Requirements**: Language interpreters/compilers installed

### Language Configuration

Each supported language has specific configuration:

```python
LANGUAGE_CONFIGS = {
    SupportedLanguage.PYTHON: {
        "extension": ".py",
        "docker_image": "python:3.11-alpine",
        "native_command": [sys.executable],
        "interpreter": "python3",
        "security_restrictions": [
            "import os", "import subprocess", "import sys",
            "__import__", "eval", "exec", "compile"
        ]
    },
    # ... other languages
}
```

## Enhanced Tools

### Core Execution

#### secure_code_executor_enhanced
Execute code with comprehensive security and monitoring.

```python
secure_code_executor_enhanced(
    code="print('Hello World!')",
    language="python",
    environment="docker",
    max_execution_time=30,
    max_memory_mb=512,
    max_cpu_percent=80.0,
    network_access=False,
    file_system_access=False
)
```

**Parameters:**
- `code`: Code to execute
- `language`: Programming language (python, javascript, bash, ruby, go, rust, java, cpp, c)
- `environment`: Execution environment (docker, native)
- `max_execution_time`: Maximum execution time in seconds
- `max_memory_mb`: Maximum memory usage in MB
- `max_cpu_percent`: Maximum CPU usage percentage
- `network_access`: Allow network access (default: False)
- `file_system_access`: Allow file system access (default: False)

### Security Validation

#### validate_code_security
Pre-execution security validation and risk assessment.

```python
validate_code_security(
    code="import os; os.listdir('/')",
    language="python",
    strict_mode=True
)
```

**Returns:**
```json
{
  "status": "success",
  "is_safe": false,
  "violations": [
    "Restricted pattern found: import os",
    "Dangerous operation detected: import\\s+os"
  ],
  "language": "python",
  "strict_mode": true
}
```

### System Information

#### list_supported_languages
Get comprehensive information about supported languages.

```python
list_supported_languages()
```

**Returns:**
```json
{
  "status": "success",
  "supported_languages": {
    "python": {
      "name": "python",
      "extension": ".py",
      "docker_image": "python:3.11-alpine",
      "interpreter": "python3",
      "has_native_support": true,
      "security_restrictions": 4
    }
  },
  "total_languages": 9,
  "docker_available": true
}
```

#### get_resource_limits_info
Get information about resource limits and configurations.

```python
get_resource_limits_info()
```

**Returns:**
```json
{
  "status": "success",
  "default_limits": {
    "max_execution_time": 30,
    "max_memory_mb": 512,
    "max_cpu_percent": 80.0,
    "max_file_size_mb": 10,
    "max_output_size_kb": 1024,
    "max_processes": 5,
    "network_access": false,
    "file_system_access": false
  },
  "recommended_limits": {
    "development": {
      "max_execution_time": 60,
      "max_memory_mb": 1024,
      "max_cpu_percent": 90.0,
      "network_access": true,
      "file_system_access": true
    },
    "production": {
      "max_execution_time": 30,
      "max_memory_mb": 512,
      "max_cpu_percent": 80.0,
      "network_access": false,
      "file_system_access": false
    }
  }
}
```

#### get_executor_stats
Get comprehensive execution statistics and metrics.

```python
get_executor_stats(include_history=True)
```

## Usage Examples

### Basic Enhanced Execution

```python
from tools.secure_code_executor_enhanced import secure_code_executor_enhanced
import json

# Execute Python code with default security
result = secure_code_executor_enhanced(
    code="print('Hello World!')\nprint(2 + 2)",
    language="python",
    environment="docker"
)

result_data = json.loads(result)
print(f"Status: {result_data['status']}")
print(f"Output: {result_data['stdout']}")
```

### Multi-Language Execution

```python
# Python execution
python_result = secure_code_executor_enhanced(
    code="print('Hello from Python!')",
    language="python"
)

# JavaScript execution
js_result = secure_code_executor_enhanced(
    code="console.log('Hello from JavaScript!');",
    language="javascript"
)

# Bash execution
bash_result = secure_code_executor_enhanced(
    code="echo 'Hello from Bash!'",
    language="bash"
)
```

### Resource-Limited Execution

```python
# Execute with strict resource limits
result = secure_code_executor_enhanced(
    code="import time; time.sleep(2); print('Done')",
    language="python",
    environment="docker",
    max_execution_time=5,
    max_memory_mb=256,
    max_cpu_percent=50.0
)
```

### Security Validation Workflow

```python
from tools.secure_code_executor_enhanced import validate_code_security, secure_code_executor_enhanced
import json

# Validate code before execution
validation = validate_code_security(
    code="import os; print(os.getcwd())",
    language="python",
    strict_mode=True
)

validation_data = json.loads(validation)

if validation_data["is_safe"]:
    # Execute if safe
    result = secure_code_executor_enhanced(
        code="import os; print(os.getcwd())",
        language="python"
    )
else:
    print(f"Security violations: {validation_data['violations']}")
```

### Development vs Production Configuration

```python
# Development configuration (more permissive)
dev_result = secure_code_executor_enhanced(
    code="import requests; print('Network access allowed')",
    language="python",
    environment="native",
    max_execution_time=60,
    max_memory_mb=1024,
    network_access=True,
    file_system_access=True
)

# Production configuration (strict security)
prod_result = secure_code_executor_enhanced(
    code="print('Secure execution')",
    language="python",
    environment="docker",
    max_execution_time=30,
    max_memory_mb=512,
    network_access=False,
    file_system_access=False
)
```

## Configuration

### Docker Configuration

```python
# Docker sandbox configuration
docker_config = {
    "base_images": {
        "python": "python:3.11-alpine",
        "javascript": "node:18-alpine",
        "bash": "alpine:latest",
        "ruby": "ruby:3.2-alpine"
    },
    "security_options": [
        "--network=none",  # No network access
        "--read-only",     # Read-only file system
        "--user=nobody",   # Unprivileged user
        "--no-new-privileges"  # Prevent privilege escalation
    ]
}
```

### Resource Limits Configuration

```python
# Resource limits for different environments
resource_configs = {
    "development": ResourceLimits(
        max_execution_time=60,
        max_memory_mb=1024,
        max_cpu_percent=90.0,
        network_access=True,
        file_system_access=True
    ),
    "testing": ResourceLimits(
        max_execution_time=10,
        max_memory_mb=256,
        max_cpu_percent=50.0,
        network_access=False,
        file_system_access=False
    ),
    "production": ResourceLimits(
        max_execution_time=30,
        max_memory_mb=512,
        max_cpu_percent=80.0,
        network_access=False,
        file_system_access=False
    )
}
```

### Security Configuration

```python
# Security validation configuration
security_config = {
    "strict_mode": True,
    "blocked_imports": [
        "os", "sys", "subprocess", "socket",
        "urllib", "requests", "http"
    ],
    "blocked_functions": [
        "eval", "exec", "compile", "__import__",
        "open", "file", "input", "raw_input"
    ],
    "max_code_size": 50000,  # 50KB
    "scan_patterns": True
}
```

## Security Features

### Sandboxing Security
- **Complete Isolation**: Docker containers provide full system isolation
- **Network Isolation**: Configurable network access with default denial
- **File System Protection**: Read-only or restricted file system access
- **User Isolation**: Code runs as unprivileged user (nobody)
- **Resource Isolation**: Containerized resource limits prevent host impact

### Code Validation Security
- **Pre-Execution Scanning**: Security validation before code execution
- **Pattern Detection**: Regex-based detection of dangerous patterns
- **Language-Specific Rules**: Tailored security rules for each language
- **Import Restrictions**: Block dangerous imports and modules
- **Function Blacklisting**: Prevent execution of dangerous functions

### Runtime Security
- **Resource Monitoring**: Real-time tracking of resource usage
- **Automatic Termination**: Kill processes exceeding limits
- **Output Sanitization**: Limit and sanitize output content
- **Error Handling**: Secure error messages without information leakage
- **Audit Logging**: Complete execution audit trail

## Performance Optimization

### Execution Performance
- **Container Reuse**: Reuse Docker containers when possible
- **Image Caching**: Cache Docker images for faster startup
- **Native Fallback**: Use native execution for trusted environments
- **Parallel Execution**: Support for concurrent code execution
- **Resource Pooling**: Pool resources for better utilization

### Memory Management
- **Memory Limits**: Strict memory usage enforcement
- **Garbage Collection**: Automatic cleanup of temporary resources
- **Container Cleanup**: Automatic removal of stopped containers
- **Resource Monitoring**: Track and optimize resource usage
- **Leak Prevention**: Prevent memory leaks in long-running processes

### Scalability Features
- **Horizontal Scaling**: Support for distributed execution
- **Load Balancing**: Distribute execution across available resources
- **Queue Management**: Handle multiple execution requests efficiently
- **Resource Scheduling**: Intelligent resource allocation
- **Auto-Scaling**: Automatic scaling based on demand

## Monitoring and Observability

### Execution Metrics
- **Success Rate**: Track execution success and failure rates
- **Performance Metrics**: Monitor execution time and resource usage
- **Language Usage**: Track usage patterns by programming language
- **Environment Usage**: Monitor Docker vs native execution usage
- **Error Analysis**: Categorize and analyze execution failures

### Security Metrics
- **Violation Detection**: Track security violations and attempts
- **Threat Analysis**: Analyze patterns in malicious code attempts
- **Access Patterns**: Monitor network and file system access attempts
- **Compliance Tracking**: Ensure compliance with security policies
- **Incident Response**: Automated response to security incidents

### Resource Metrics
- **Resource Utilization**: Track CPU, memory, and disk usage
- **Capacity Planning**: Monitor and predict resource needs
- **Performance Trends**: Analyze performance trends over time
- **Bottleneck Identification**: Identify and resolve performance bottlenecks
- **Cost Optimization**: Optimize resource usage for cost efficiency

## Best Practices

### Security Best Practices
1. **Use Docker Environment**: Always use Docker for untrusted code
2. **Validate Before Execution**: Run security validation on all code
3. **Minimal Permissions**: Use least privilege principle
4. **Regular Updates**: Keep Docker images and dependencies updated
5. **Monitor Executions**: Implement comprehensive monitoring and alerting

### Performance Best Practices
1. **Set Appropriate Limits**: Configure resource limits based on use case
2. **Use Native for Trusted Code**: Use native execution for performance-critical trusted code
3. **Cache Docker Images**: Pre-pull and cache Docker images
4. **Monitor Resource Usage**: Track and optimize resource consumption
5. **Implement Timeouts**: Always set execution timeouts

### Operational Best Practices
1. **Regular Maintenance**: Clean up old containers and images
2. **Backup Configurations**: Backup security and resource configurations
3. **Test Configurations**: Regularly test security and resource limits
4. **Document Policies**: Maintain clear security and usage policies
5. **Train Users**: Educate users on secure coding practices

## Troubleshooting

### Common Issues

#### Docker Not Available
- **Cause**: Docker not installed or not accessible
- **Solution**: Install Docker and ensure proper permissions
- **Workaround**: Use native execution environment

#### Security Violations
- **Cause**: Code contains restricted patterns or operations
- **Solution**: Review and modify code to remove security violations
- **Prevention**: Use security validation before execution

#### Resource Limit Exceeded
- **Cause**: Code exceeds configured resource limits
- **Solution**: Optimize code or increase resource limits
- **Prevention**: Set appropriate limits for use case

#### Language Not Supported
- **Cause**: Requested language not configured or available
- **Solution**: Check supported languages and install required interpreters
- **Workaround**: Use alternative supported language

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enhanced Secure Code Executor will provide detailed logs
result = secure_code_executor_enhanced(
    code="print('debug test')",
    language="python"
)
```

### Recovery Procedures

#### Container Cleanup
```bash
# Remove all stopped containers
docker container prune -f

# Remove unused images
docker image prune -f

# Remove all unused resources
docker system prune -f
```

#### Resource Recovery
```python
# Check system resources
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"Disk: {psutil.disk_usage('/').percent}%")
```

## Future Enhancements

### Planned Features
1. **Advanced Sandboxing**: Integration with additional container runtimes
2. **GPU Support**: Support for GPU-accelerated code execution
3. **Distributed Execution**: Multi-node execution for large workloads
4. **Advanced Monitoring**: Machine learning-based anomaly detection
5. **Custom Images**: Support for custom Docker images and environments

### Integration Opportunities
1. **Cloud Platforms**: Integration with AWS Lambda, Google Cloud Functions
2. **Kubernetes**: Native Kubernetes job execution
3. **CI/CD Pipelines**: Integration with continuous integration systems
4. **IDE Plugins**: Direct integration with development environments
5. **API Gateways**: RESTful API for remote code execution

## Conclusion

Enhanced Secure Code Executor represents a significant advancement in secure code execution, providing:

- **Enterprise-Grade Sandboxing** with Docker containerization for complete isolation
- **Comprehensive Multi-Language Support** for 9+ programming languages
- **Advanced Resource Management** with comprehensive limits and monitoring

These improvements make AgentK suitable for production environments requiring secure, flexible, and reliable code execution with comprehensive security, performance monitoring, and multi-language support.