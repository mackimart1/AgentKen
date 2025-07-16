"""
Enhanced Secure Code Executor with Sandboxing, Multi-Language Support, and Resource Limits

Key Enhancements:
1. Sandboxing: Docker containerization for secure system-level isolation
2. Multi-Language Support: Python, JavaScript, Bash, Ruby, and more
3. Resource Limits: CPU time, memory usage, and execution timeout constraints

This enhanced version provides enterprise-grade security and flexibility for code execution.
"""

import subprocess
import json
import sys
import tempfile
import os
import time
import threading
import psutil
import shutil
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionEnvironment(Enum):
    """Execution environment types."""
    NATIVE = "native"
    DOCKER = "docker"
    RESTRICTED = "restricted"


class SupportedLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    C = "c"


@dataclass
class ResourceLimits:
    """Resource limits for code execution."""
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 512  # MB
    max_cpu_percent: float = 80.0  # percentage
    max_file_size_mb: int = 10  # MB
    max_output_size_kb: int = 1024  # KB
    max_processes: int = 5
    network_access: bool = False
    file_system_access: bool = False


@dataclass
class ExecutionResult:
    """Result of code execution with comprehensive metadata."""
    status: str
    stdout: Optional[str]
    stderr: Optional[str]
    error: Optional[str]
    message: str
    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    exit_code: Optional[int]
    language: str
    environment: str
    resource_limits: Dict[str, Any]
    security_violations: List[str]
    timestamp: str
    execution_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LanguageConfig:
    """Configuration for different programming languages."""
    
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
        SupportedLanguage.JAVASCRIPT: {
            "extension": ".js",
            "docker_image": "node:18-alpine",
            "native_command": ["node"],
            "interpreter": "node",
            "security_restrictions": [
                "require('fs')", "require('child_process')", "require('os')",
                "eval(", "Function(", "setTimeout", "setInterval"
            ]
        },
        SupportedLanguage.BASH: {
            "extension": ".sh",
            "docker_image": "alpine:latest",
            "native_command": ["bash"],
            "interpreter": "bash",
            "security_restrictions": [
                "rm ", "sudo ", "su ", "chmod ", "chown ",
                "wget ", "curl ", "nc ", "netcat"
            ]
        },
        SupportedLanguage.RUBY: {
            "extension": ".rb",
            "docker_image": "ruby:3.2-alpine",
            "native_command": ["ruby"],
            "interpreter": "ruby",
            "security_restrictions": [
                "system(", "`", "exec(", "eval(", "require 'open3'"
            ]
        },
        SupportedLanguage.GO: {
            "extension": ".go",
            "docker_image": "golang:1.21-alpine",
            "native_command": ["go", "run"],
            "interpreter": "go",
            "security_restrictions": [
                "os/exec", "syscall", "unsafe", "net/http"
            ]
        },
        SupportedLanguage.RUST: {
            "extension": ".rs",
            "docker_image": "rust:1.75-alpine",
            "native_command": ["rustc", "--edition", "2021"],
            "interpreter": "rustc",
            "security_restrictions": [
                "std::process", "std::fs", "unsafe"
            ]
        },
        SupportedLanguage.JAVA: {
            "extension": ".java",
            "docker_image": "openjdk:17-alpine",
            "native_command": ["javac"],
            "interpreter": "java",
            "security_restrictions": [
                "Runtime.getRuntime()", "ProcessBuilder", "System.exit"
            ]
        },
        SupportedLanguage.CPP: {
            "extension": ".cpp",
            "docker_image": "gcc:latest",
            "native_command": ["g++", "-o"],
            "interpreter": "g++",
            "security_restrictions": [
                "#include <cstdlib>", "system(", "exec"
            ]
        },
        SupportedLanguage.C: {
            "extension": ".c",
            "docker_image": "gcc:latest",
            "native_command": ["gcc", "-o"],
            "interpreter": "gcc",
            "security_restrictions": [
                "#include <stdlib.h>", "system(", "exec"
            ]
        }
    }
    
    @classmethod
    def get_config(cls, language: SupportedLanguage) -> Dict[str, Any]:
        return cls.LANGUAGE_CONFIGS.get(language, {})
    
    @classmethod
    def is_supported(cls, language: str) -> bool:
        try:
            SupportedLanguage(language.lower())
            return True
        except ValueError:
            return False


class SecurityValidator:
    """Validates code for security risks before execution."""
    
    def __init__(self):
        self.violation_patterns = {
            "file_operations": [
                r"open\s*\(", r"file\s*\(", r"with\s+open",
                r"os\.remove", r"os\.unlink", r"shutil\."
            ],
            "network_operations": [
                r"urllib", r"requests", r"socket", r"http",
                r"fetch\s*\(", r"XMLHttpRequest"
            ],
            "system_operations": [
                r"os\.system", r"subprocess", r"exec\s*\(",
                r"eval\s*\(", r"__import__"
            ],
            "dangerous_imports": [
                r"import\s+os", r"import\s+sys", r"import\s+subprocess",
                r"from\s+os", r"from\s+sys"
            ]
        }
    
    def validate_code(self, code: str, language: SupportedLanguage, 
                     resource_limits: ResourceLimits) -> Tuple[bool, List[str]]:
        """Validate code for security violations."""
        violations = []
        
        # Get language-specific restrictions
        config = LanguageConfig.get_config(language)
        restrictions = config.get("security_restrictions", [])
        
        # Check for language-specific violations
        for restriction in restrictions:
            if restriction in code:
                violations.append(f"Restricted pattern found: {restriction}")
        
        # Check for general security violations
        import re
        for category, patterns in self.violation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    if category == "file_operations" and not resource_limits.file_system_access:
                        violations.append(f"File system access not allowed: {pattern}")
                    elif category == "network_operations" and not resource_limits.network_access:
                        violations.append(f"Network access not allowed: {pattern}")
                    elif category in ["system_operations", "dangerous_imports"]:
                        violations.append(f"Dangerous operation detected: {pattern}")
        
        # Check code length
        if len(code) > 50000:  # 50KB limit
            violations.append("Code too long (>50KB)")
        
        return len(violations) == 0, violations


class DockerSandbox:
    """Docker-based sandboxing for secure code execution."""
    
    def __init__(self):
        self.docker_available = self._check_docker_availability()
    
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available and accessible."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def execute_in_container(self, code: str, language: SupportedLanguage,
                           resource_limits: ResourceLimits) -> ExecutionResult:
        """Execute code in a Docker container."""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        if not self.docker_available:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error="Docker not available",
                message="Docker is not installed or accessible",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language.value,
                environment=ExecutionEnvironment.DOCKER.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
        
        config = LanguageConfig.get_config(language)
        docker_image = config.get("docker_image")
        extension = config.get("extension")
        
        # Create temporary directory for code
        temp_dir = Path(tempfile.mkdtemp())
        code_file = temp_dir / f"code{extension}"
        
        try:
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Prepare Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",
                "--network", "none" if not resource_limits.network_access else "bridge",
                "--memory", f"{resource_limits.max_memory_mb}m",
                "--cpus", str(resource_limits.max_cpu_percent / 100),
                "--pids-limit", str(resource_limits.max_processes),
                "--read-only" if not resource_limits.file_system_access else "--tmpfs", "/tmp",
                "--tmpfs", "/var/tmp",
                "--user", "nobody",
                "--workdir", "/workspace",
                "-v", f"{temp_dir}:/workspace:ro",
                "--timeout", str(resource_limits.max_execution_time),
                docker_image
            ]
            
            # Add language-specific execution command
            if language == SupportedLanguage.PYTHON:
                docker_cmd.extend(["python3", f"code{extension}"])
            elif language == SupportedLanguage.JAVASCRIPT:
                docker_cmd.extend(["node", f"code{extension}"])
            elif language == SupportedLanguage.BASH:
                docker_cmd.extend(["sh", f"code{extension}"])
            elif language == SupportedLanguage.RUBY:
                docker_cmd.extend(["ruby", f"code{extension}"])
            elif language == SupportedLanguage.GO:
                docker_cmd.extend(["go", "run", f"code{extension}"])
            elif language in [SupportedLanguage.C, SupportedLanguage.CPP]:
                # Compile and run
                docker_cmd.extend(["sh", "-c", 
                    f"gcc -o /tmp/program code{extension} && /tmp/program"])
            elif language == SupportedLanguage.JAVA:
                # Compile and run Java
                docker_cmd.extend(["sh", "-c", 
                    f"javac code{extension} && java $(basename code{extension} .java)"])
            
            # Execute in Docker
            process = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=resource_limits.max_execution_time + 5  # Extra buffer
            )
            
            execution_time = time.time() - start_time
            
            # Truncate output if too large
            max_output = resource_limits.max_output_size_kb * 1024
            stdout = process.stdout[:max_output] if process.stdout else None
            stderr = process.stderr[:max_output] if process.stderr else None
            
            if len(process.stdout or "") > max_output:
                stdout += "\n[Output truncated - exceeded size limit]"
            
            return ExecutionResult(
                status="success" if process.returncode == 0 else "failure",
                stdout=stdout,
                stderr=stderr,
                error=stderr if process.returncode != 0 else None,
                message="Code executed in Docker container",
                execution_time=execution_time,
                memory_used_mb=0,  # Docker stats would need separate call
                cpu_percent=0,     # Docker stats would need separate call
                exit_code=process.returncode,
                language=language.value,
                environment=ExecutionEnvironment.DOCKER.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error="Execution timeout",
                message=f"Code execution timed out after {resource_limits.max_execution_time} seconds",
                execution_time=resource_limits.max_execution_time,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language.value,
                environment=ExecutionEnvironment.DOCKER.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
        except Exception as e:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error=str(e),
                message=f"Docker execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language.value,
                environment=ExecutionEnvironment.DOCKER.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)


class NativeSandbox:
    """Native execution with resource monitoring and limits."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def execute_native(self, code: str, language: SupportedLanguage,
                      resource_limits: ResourceLimits) -> ExecutionResult:
        """Execute code natively with resource monitoring."""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        config = LanguageConfig.get_config(language)
        extension = config.get("extension")
        native_command = config.get("native_command", [])
        
        if not native_command:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error="Language not supported for native execution",
                message=f"Native execution not configured for {language.value}",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language.value,
                environment=ExecutionEnvironment.NATIVE.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
        
        # Create temporary file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=extension, delete=False
            ) as f:
                temp_file = f.name
                f.write(code)
            
            # Prepare command
            if language == SupportedLanguage.PYTHON:
                cmd = [sys.executable, temp_file]
            elif language == SupportedLanguage.JAVASCRIPT:
                cmd = ["node", temp_file]
            elif language == SupportedLanguage.BASH:
                cmd = ["bash", temp_file]
            elif language == SupportedLanguage.RUBY:
                cmd = ["ruby", temp_file]
            elif language == SupportedLanguage.GO:
                cmd = ["go", "run", temp_file]
            else:
                cmd = native_command + [temp_file]
            
            # Execute with monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor resources
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process, resource_limits)
            )
            monitor_thread.start()
            
            try:
                stdout, stderr = process.communicate(
                    timeout=resource_limits.max_execution_time
                )
                monitor_thread.join(timeout=1)
                
                execution_time = time.time() - start_time
                
                # Get process info if still available
                memory_used = 0
                cpu_percent = 0
                try:
                    if process.pid and psutil.pid_exists(process.pid):
                        proc_info = psutil.Process(process.pid)
                        memory_used = proc_info.memory_info().rss / 1024 / 1024  # MB
                        cpu_percent = proc_info.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Truncate output if needed
                max_output = resource_limits.max_output_size_kb * 1024
                if len(stdout) > max_output:
                    stdout = stdout[:max_output] + "\n[Output truncated]"
                if len(stderr) > max_output:
                    stderr = stderr[:max_output] + "\n[Error output truncated]"
                
                return ExecutionResult(
                    status="success" if process.returncode == 0 else "failure",
                    stdout=stdout,
                    stderr=stderr,
                    error=stderr if process.returncode != 0 else None,
                    message="Code executed natively with monitoring",
                    execution_time=execution_time,
                    memory_used_mb=memory_used,
                    cpu_percent=cpu_percent,
                    exit_code=process.returncode,
                    language=language.value,
                    environment=ExecutionEnvironment.NATIVE.value,
                    resource_limits=asdict(resource_limits),
                    security_violations=[],
                    timestamp=datetime.now().isoformat(),
                    execution_id=execution_id
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return ExecutionResult(
                    status="failure",
                    stdout=None,
                    stderr=None,
                    error="Execution timeout",
                    message=f"Execution timed out after {resource_limits.max_execution_time} seconds",
                    execution_time=resource_limits.max_execution_time,
                    memory_used_mb=0,
                    cpu_percent=0,
                    exit_code=None,
                    language=language.value,
                    environment=ExecutionEnvironment.NATIVE.value,
                    resource_limits=asdict(resource_limits),
                    security_violations=[],
                    timestamp=datetime.now().isoformat(),
                    execution_id=execution_id
                )
                
        except Exception as e:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error=str(e),
                message=f"Native execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language.value,
                environment=ExecutionEnvironment.NATIVE.value,
                resource_limits=asdict(resource_limits),
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id
            )
        finally:
            # Cleanup
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _monitor_process(self, process: subprocess.Popen, limits: ResourceLimits):
        """Monitor process resource usage and enforce limits."""
        try:
            if not process.pid:
                return
            
            proc = psutil.Process(process.pid)
            
            while process.poll() is None:
                try:
                    # Check memory usage
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    if memory_mb > limits.max_memory_mb:
                        logger.warning(f"Process {process.pid} exceeded memory limit")
                        process.kill()
                        break
                    
                    # Check CPU usage
                    cpu_percent = proc.cpu_percent()
                    if cpu_percent > limits.max_cpu_percent:
                        logger.warning(f"Process {process.pid} exceeded CPU limit")
                        # Don't kill immediately for CPU, just log
                    
                    # Check number of child processes
                    children = proc.children(recursive=True)
                    if len(children) > limits.max_processes:
                        logger.warning(f"Process {process.pid} exceeded process limit")
                        process.kill()
                        break
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")


class EnhancedSecureCodeExecutor:
    """Enhanced secure code executor with sandboxing, multi-language support, and resource limits."""
    
    def __init__(self):
        self.docker_sandbox = DockerSandbox()
        self.native_sandbox = NativeSandbox()
        self.security_validator = SecurityValidator()
        self.execution_history: List[ExecutionResult] = []
    
    def execute_code(self, code: str, language: str = "python",
                    environment: str = "docker",
                    resource_limits: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute code with enhanced security and monitoring."""
        
        # Validate language
        if not LanguageConfig.is_supported(language):
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error="Unsupported language",
                message=f"Language '{language}' is not supported",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language,
                environment=environment,
                resource_limits={},
                security_violations=[],
                timestamp=datetime.now().isoformat(),
                execution_id=str(uuid.uuid4())
            )
        
        lang_enum = SupportedLanguage(language.lower())
        
        # Set up resource limits
        if resource_limits:
            limits = ResourceLimits(**resource_limits)
        else:
            limits = ResourceLimits()
        
        # Validate code security
        is_safe, violations = self.security_validator.validate_code(
            code, lang_enum, limits
        )
        
        if not is_safe:
            return ExecutionResult(
                status="failure",
                stdout=None,
                stderr=None,
                error="Security violations detected",
                message=f"Code contains security violations: {', '.join(violations)}",
                execution_time=0,
                memory_used_mb=0,
                cpu_percent=0,
                exit_code=None,
                language=language,
                environment=environment,
                resource_limits=asdict(limits),
                security_violations=violations,
                timestamp=datetime.now().isoformat(),
                execution_id=str(uuid.uuid4())
            )
        
        # Execute based on environment
        if environment.lower() == "docker":
            result = self.docker_sandbox.execute_in_container(code, lang_enum, limits)
        else:
            result = self.native_sandbox.execute_native(code, lang_enum, limits)
        
        # Store execution history
        self.execution_history.append(result)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history if r.status == "success"])
        
        languages = {}
        environments = {}
        
        for result in self.execution_history:
            languages[result.language] = languages.get(result.language, 0) + 1
            environments[result.environment] = environments.get(result.environment, 0) + 1
        
        avg_execution_time = sum(r.execution_time for r in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total * 100,
            "languages_used": languages,
            "environments_used": environments,
            "average_execution_time": avg_execution_time,
            "docker_available": self.docker_sandbox.docker_available
        }


# Global enhanced executor instance
_enhanced_executor = EnhancedSecureCodeExecutor()


# Input models for the enhanced tools
class ExecuteCodeInput(BaseModel):
    code: str = Field(description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    environment: str = Field(default="docker", description="Execution environment (docker/native)")
    max_execution_time: int = Field(default=30, description="Maximum execution time in seconds")
    max_memory_mb: int = Field(default=512, description="Maximum memory usage in MB")
    max_cpu_percent: float = Field(default=80.0, description="Maximum CPU usage percentage")
    network_access: bool = Field(default=False, description="Allow network access")
    file_system_access: bool = Field(default=False, description="Allow file system access")


class GetStatsInput(BaseModel):
    include_history: bool = Field(default=False, description="Include execution history")


class ValidateCodeInput(BaseModel):
    code: str = Field(description="Code to validate")
    language: str = Field(default="python", description="Programming language")
    strict_mode: bool = Field(default=True, description="Enable strict security validation")


# Enhanced secure code executor tools
@tool(args_schema=ExecuteCodeInput)
def secure_code_executor_enhanced(
    code: str,
    language: str = "python",
    environment: str = "docker",
    max_execution_time: int = 30,
    max_memory_mb: int = 512,
    max_cpu_percent: float = 80.0,
    network_access: bool = False,
    file_system_access: bool = False
) -> str:
    """
    Execute code in a secure sandboxed environment with comprehensive resource limits and multi-language support.
    
    Args:
        code: Code to execute
        language: Programming language (python, javascript, bash, ruby, go, rust, java, cpp, c)
        environment: Execution environment (docker for containerized, native for local)
        max_execution_time: Maximum execution time in seconds
        max_memory_mb: Maximum memory usage in MB
        max_cpu_percent: Maximum CPU usage percentage
        network_access: Allow network access
        file_system_access: Allow file system access
    
    Returns:
        JSON string with execution results and metadata
    """
    try:
        resource_limits = {
            "max_execution_time": max_execution_time,
            "max_memory_mb": max_memory_mb,
            "max_cpu_percent": max_cpu_percent,
            "network_access": network_access,
            "file_system_access": file_system_access
        }
        
        result = _enhanced_executor.execute_code(
            code=code,
            language=language,
            environment=environment,
            resource_limits=resource_limits
        )
        
        return json.dumps(result.to_dict(), indent=2)
        
    except Exception as e:
        error_result = ExecutionResult(
            status="failure",
            stdout=None,
            stderr=None,
            error=str(e),
            message=f"Enhanced execution failed: {str(e)}",
            execution_time=0,
            memory_used_mb=0,
            cpu_percent=0,
            exit_code=None,
            language=language,
            environment=environment,
            resource_limits={},
            security_violations=[],
            timestamp=datetime.now().isoformat(),
            execution_id=str(uuid.uuid4())
        )
        return json.dumps(error_result.to_dict(), indent=2)


@tool(args_schema=ValidateCodeInput)
def validate_code_security(code: str, language: str = "python", strict_mode: bool = True) -> str:
    """
    Validate code for security risks before execution.
    
    Args:
        code: Code to validate
        language: Programming language
        strict_mode: Enable strict security validation
    
    Returns:
        JSON string with validation results
    """
    try:
        if not LanguageConfig.is_supported(language):
            return json.dumps({
                "status": "failure",
                "message": f"Language '{language}' is not supported",
                "is_safe": False,
                "violations": [f"Unsupported language: {language}"]
            })
        
        lang_enum = SupportedLanguage(language.lower())
        limits = ResourceLimits(
            network_access=not strict_mode,
            file_system_access=not strict_mode
        )
        
        validator = SecurityValidator()
        is_safe, violations = validator.validate_code(code, lang_enum, limits)
        
        return json.dumps({
            "status": "success",
            "is_safe": is_safe,
            "violations": violations,
            "language": language,
            "strict_mode": strict_mode,
            "validation_timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Validation failed: {str(e)}",
            "is_safe": False,
            "violations": [str(e)]
        })


@tool(args_schema=GetStatsInput)
def get_executor_stats(include_history: bool = False) -> str:
    """
    Get comprehensive statistics about code execution.
    
    Args:
        include_history: Include detailed execution history
    
    Returns:
        JSON string with execution statistics
    """
    try:
        stats = _enhanced_executor.get_execution_stats()
        
        if include_history:
            stats["execution_history"] = [
                result.to_dict() for result in _enhanced_executor.execution_history[-10:]
            ]
        
        stats["supported_languages"] = [lang.value for lang in SupportedLanguage]
        stats["available_environments"] = [env.value for env in ExecutionEnvironment]
        
        return json.dumps({
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get statistics: {str(e)}"
        })


@tool
def list_supported_languages() -> str:
    """
    List all supported programming languages with their configurations.
    
    Returns:
        JSON string with supported languages and their details
    """
    try:
        languages = {}
        
        for lang in SupportedLanguage:
            config = LanguageConfig.get_config(lang)
            languages[lang.value] = {
                "name": lang.value,
                "extension": config.get("extension", ""),
                "docker_image": config.get("docker_image", ""),
                "interpreter": config.get("interpreter", ""),
                "has_native_support": bool(config.get("native_command")),
                "security_restrictions": len(config.get("security_restrictions", []))
            }
        
        return json.dumps({
            "status": "success",
            "supported_languages": languages,
            "total_languages": len(languages),
            "docker_available": _enhanced_executor.docker_sandbox.docker_available
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to list languages: {str(e)}"
        })


@tool
def get_resource_limits_info() -> str:
    """
    Get information about available resource limits and their defaults.
    
    Returns:
        JSON string with resource limits information
    """
    try:
        default_limits = ResourceLimits()
        
        return json.dumps({
            "status": "success",
            "default_limits": asdict(default_limits),
            "limit_descriptions": {
                "max_execution_time": "Maximum execution time in seconds",
                "max_memory_mb": "Maximum memory usage in MB",
                "max_cpu_percent": "Maximum CPU usage percentage",
                "max_file_size_mb": "Maximum file size in MB",
                "max_output_size_kb": "Maximum output size in KB",
                "max_processes": "Maximum number of processes",
                "network_access": "Allow network access",
                "file_system_access": "Allow file system access"
            },
            "recommended_limits": {
                "development": {
                    "max_execution_time": 60,
                    "max_memory_mb": 1024,
                    "max_cpu_percent": 90.0,
                    "network_access": True,
                    "file_system_access": True
                },
                "production": {
                    "max_execution_time": 30,
                    "max_memory_mb": 512,
                    "max_cpu_percent": 80.0,
                    "network_access": False,
                    "file_system_access": False
                },
                "testing": {
                    "max_execution_time": 10,
                    "max_memory_mb": 256,
                    "max_cpu_percent": 50.0,
                    "network_access": False,
                    "file_system_access": False
                }
            }
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get resource limits info: {str(e)}"
        })


# Export enhanced executor for direct use
__all__ = [
    "EnhancedSecureCodeExecutor",
    "secure_code_executor_enhanced",
    "validate_code_security",
    "get_executor_stats",
    "list_supported_languages",
    "get_resource_limits_info"
]