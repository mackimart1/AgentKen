"""
Enhanced Tool Maker Agent: Advanced tool developer with validation, documentation, and lifecycle management.

Key Enhancements:
1. Tool Validation: Sandboxed test environment for functionality and safety validation
2. Tool Documentation: Auto-generation of comprehensive usage documentation
3. Tool Deprecation: Complete lifecycle management with deprecation process

This enhanced version provides robust tool development with comprehensive quality assurance.
"""

from typing import Literal, Optional, Dict, Any, List, Tuple
import os
import sys
import time
import re
import json
import ast
import subprocess
import tempfile
import shutil
import logging
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from requests.exceptions import HTTPError, RequestException

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import utils
import config

# Setup enhanced logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize memory manager
try:
    import memory_manager
    memory_manager_instance = memory_manager.MemoryManager()
    MEMORY_AVAILABLE = True
except ImportError:
    logger.warning("Memory manager not available")
    memory_manager_instance = None
    MEMORY_AVAILABLE = False


class ToolStatus(Enum):
    """Tool lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    OBSOLETE = "obsolete"
    EXPERIMENTAL = "experimental"
    BETA = "beta"


class ValidationResult(Enum):
    """Tool validation result."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ToolMetadata:
    """Enhanced tool metadata."""
    name: str
    version: str
    status: ToolStatus
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    removal_date: Optional[datetime] = None
    deprecation_reason: str = ""
    replacement_tool: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "dependencies": self.dependencies,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "removal_date": self.removal_date.isoformat() if self.removal_date else None,
            "deprecation_reason": self.deprecation_reason,
            "replacement_tool": self.replacement_tool
        }


@dataclass
class ValidationReport:
    """Tool validation report."""
    tool_name: str
    timestamp: datetime
    overall_result: ValidationResult
    safety_score: float  # 0-100
    functionality_score: float  # 0-100
    performance_score: float  # 0-100
    security_checks: List[Dict[str, Any]] = field(default_factory=list)
    functionality_tests: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolValidator:
    """Sandboxed tool validation system."""
    
    def __init__(self, sandbox_dir: str = "sandbox"):
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(exist_ok=True)
        self.validation_history: List[ValidationReport] = []
    
    def validate_tool(self, tool_path: str, tool_name: str) -> ValidationReport:
        """Comprehensive tool validation in sandboxed environment."""
        logger.info(f"Starting validation for tool: {tool_name}")
        
        report = ValidationReport(
            tool_name=tool_name,
            timestamp=datetime.now(),
            overall_result=ValidationResult.FAILED,
            safety_score=0.0,
            functionality_score=0.0,
            performance_score=0.0
        )
        
        try:
            # Create isolated sandbox
            sandbox_path = self._create_sandbox(tool_path, tool_name)
            
            # Run validation checks
            self._validate_security(sandbox_path, tool_name, report)
            self._validate_functionality(sandbox_path, tool_name, report)
            self._validate_performance(sandbox_path, tool_name, report)
            
            # Calculate overall score and result
            overall_score = (report.safety_score + report.functionality_score + report.performance_score) / 3
            
            if overall_score >= 80 and not report.errors:
                report.overall_result = ValidationResult.PASSED
            elif overall_score >= 60 and len(report.errors) <= 1:
                report.overall_result = ValidationResult.WARNING
            else:
                report.overall_result = ValidationResult.FAILED
            
            # Cleanup sandbox
            self._cleanup_sandbox(sandbox_path)
            
        except Exception as e:
            report.errors.append(f"Validation failed: {str(e)}")
            logger.error(f"Tool validation error: {e}", exc_info=True)
        
        self.validation_history.append(report)
        return report
    
    def _create_sandbox(self, tool_path: str, tool_name: str) -> Path:
        """Create isolated sandbox environment."""
        sandbox_path = self.sandbox_dir / f"sandbox_{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sandbox_path.mkdir(exist_ok=True)
        
        # Copy tool file to sandbox
        tool_file = sandbox_path / f"{tool_name}.py"
        shutil.copy2(tool_path, tool_file)
        
        # Create minimal requirements
        requirements_file = sandbox_path / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write("langchain-core\npydantic\n")
        
        return sandbox_path
    
    def _validate_security(self, sandbox_path: Path, tool_name: str, report: ValidationReport):
        """Validate tool security and safety."""
        tool_file = sandbox_path / f"{tool_name}.py"
        
        try:
            with open(tool_file, 'r') as f:
                code = f.read()
            
            # Parse AST for security analysis
            tree = ast.parse(code)
            
            security_score = 100.0
            security_checks = []
            
            # Check for dangerous imports
            dangerous_imports = ['os', 'subprocess', 'sys', 'eval', 'exec', 'compile']
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            security_score -= 15
                            security_checks.append({
                                "check": "dangerous_import",
                                "result": "warning",
                                "details": f"Potentially dangerous import: {alias.name}",
                                "line": node.lineno
                            })
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        security_score -= 15
                        security_checks.append({
                            "check": "dangerous_import",
                            "result": "warning",
                            "details": f"Potentially dangerous import: {node.module}",
                            "line": node.lineno
                        })
            
            # Check for dangerous function calls
            dangerous_calls = ['eval', 'exec', 'compile', '__import__']
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in dangerous_calls:
                        security_score -= 25
                        security_checks.append({
                            "check": "dangerous_call",
                            "result": "error",
                            "details": f"Dangerous function call: {node.func.id}",
                            "line": node.lineno
                        })
                        report.errors.append(f"Dangerous function call: {node.func.id} at line {node.lineno}")
            
            # Check for file system operations
            file_operations = ['open', 'file', 'write', 'read']
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in file_operations:
                        security_checks.append({
                            "check": "file_operation",
                            "result": "info",
                            "details": f"File operation detected: {node.func.id}",
                            "line": node.lineno
                        })
            
            # Check for network operations
            if any(keyword in code.lower() for keyword in ['requests', 'urllib', 'http', 'socket']):
                security_checks.append({
                    "check": "network_operation",
                    "result": "info",
                    "details": "Network operations detected",
                    "line": 0
                })
            
            report.safety_score = max(0, security_score)
            report.security_checks = security_checks
            
            if security_score < 50:
                report.errors.append("Tool failed security validation")
            elif security_score < 80:
                report.warnings.append("Tool has security concerns")
            
        except Exception as e:
            report.errors.append(f"Security validation failed: {str(e)}")
            report.safety_score = 0
    
    def _validate_functionality(self, sandbox_path: Path, tool_name: str, report: ValidationReport):
        """Validate tool functionality."""
        tool_file = sandbox_path / f"{tool_name}.py"
        
        try:
            # Test basic import
            import_test = {
                "test": "import_test",
                "result": "failed",
                "details": "",
                "execution_time": 0
            }
            
            start_time = time.time()
            try:
                # Use subprocess to test import in isolation
                result = subprocess.run([
                    sys.executable, "-c", 
                    f"import sys; sys.path.insert(0, '{sandbox_path}'); import {tool_name}"
                ], capture_output=True, text=True, timeout=10, cwd=sandbox_path)
                
                if result.returncode == 0:
                    import_test["result"] = "passed"
                    import_test["details"] = "Tool imports successfully"
                else:
                    import_test["details"] = f"Import failed: {result.stderr}"
                    report.errors.append(f"Tool import failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                import_test["details"] = "Import test timed out"
                report.errors.append("Tool import timed out")
            except Exception as e:
                import_test["details"] = f"Import test error: {str(e)}"
                report.errors.append(f"Import test error: {str(e)}")
            
            import_test["execution_time"] = time.time() - start_time
            report.functionality_tests.append(import_test)
            
            # Test tool decorator and structure
            structure_test = {
                "test": "structure_test",
                "result": "failed",
                "details": "",
                "execution_time": 0
            }
            
            start_time = time.time()
            try:
                with open(tool_file, 'r') as f:
                    code = f.read()
                
                # Check for @tool decorator
                if "@tool" in code:
                    structure_test["result"] = "passed"
                    structure_test["details"] = "Tool decorator found"
                else:
                    structure_test["details"] = "Missing @tool decorator"
                    report.warnings.append("Tool missing @tool decorator")
                
                # Check for docstring
                if '"""' in code or "'''" in code:
                    structure_test["details"] += ", docstring present"
                else:
                    structure_test["details"] += ", missing docstring"
                    report.warnings.append("Tool missing docstring")
                
            except Exception as e:
                structure_test["details"] = f"Structure test error: {str(e)}"
            
            structure_test["execution_time"] = time.time() - start_time
            report.functionality_tests.append(structure_test)
            
            # Calculate functionality score
            passed_tests = sum(1 for test in report.functionality_tests if test["result"] == "passed")
            total_tests = len(report.functionality_tests)
            report.functionality_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
        except Exception as e:
            report.errors.append(f"Functionality validation failed: {str(e)}")
            report.functionality_score = 0
    
    def _validate_performance(self, sandbox_path: Path, tool_name: str, report: ValidationReport):
        """Validate tool performance."""
        try:
            # Measure file size
            tool_file = sandbox_path / f"{tool_name}.py"
            file_size = tool_file.stat().st_size
            
            # Measure complexity (lines of code)
            with open(tool_file, 'r') as f:
                lines = f.readlines()
            
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # Performance scoring
            performance_score = 100.0
            
            # File size penalty (over 10KB)
            if file_size > 10240:
                performance_score -= 10
                report.warnings.append(f"Large file size: {file_size} bytes")
            
            # Complexity penalty (over 200 lines)
            if code_lines > 200:
                performance_score -= 15
                report.warnings.append(f"High complexity: {code_lines} lines of code")
            
            report.performance_metrics = {
                "file_size_bytes": file_size,
                "lines_of_code": code_lines,
                "estimated_memory_usage": file_size * 2,  # Rough estimate
                "complexity_score": min(100, code_lines / 2)
            }
            
            report.performance_score = max(0, performance_score)
            
        except Exception as e:
            report.errors.append(f"Performance validation failed: {str(e)}")
            report.performance_score = 0
    
    def _cleanup_sandbox(self, sandbox_path: Path):
        """Clean up sandbox environment."""
        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox {sandbox_path}: {e}")


class ToolDocumentationGenerator:
    """Automatic tool documentation generator."""
    
    def __init__(self, docs_dir: str = "docs/tools"):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_documentation(self, tool_path: str, tool_name: str, metadata: ToolMetadata) -> str:
        """Generate comprehensive documentation for a tool."""
        logger.info(f"Generating documentation for tool: {tool_name}")
        
        try:
            # Analyze tool code
            analysis = self._analyze_tool_code(tool_path)
            
            # Generate documentation content
            doc_content = self._create_documentation_content(tool_name, metadata, analysis)
            
            # Save documentation
            doc_file = self.docs_dir / f"{tool_name}.md"
            with open(doc_file, 'w') as f:
                f.write(doc_content)
            
            logger.info(f"Documentation generated: {doc_file}")
            return str(doc_file)
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}", exc_info=True)
            return ""
    
    def _analyze_tool_code(self, tool_path: str) -> Dict[str, Any]:
        """Analyze tool code to extract documentation information."""
        analysis = {
            "function_name": "",
            "docstring": "",
            "parameters": [],
            "return_type": "",
            "imports": [],
            "examples": [],
            "edge_cases": []
        }
        
        try:
            with open(tool_path, 'r') as f:
                code = f.read()
            
            # Parse AST
            tree = ast.parse(code)
            
            # Find the main tool function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for function with @tool decorator
                    has_tool_decorator = any(
                        isinstance(decorator, ast.Name) and decorator.id == 'tool'
                        for decorator in node.decorator_list
                    )
                    
                    if has_tool_decorator:
                        analysis["function_name"] = node.name
                        
                        # Extract docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant)):
                            analysis["docstring"] = node.body[0].value.value
                        
                        # Extract parameters
                        for arg in node.args.args:
                            param_info = {"name": arg.arg, "type": "Any", "default": None}
                            
                            # Try to get type annotation
                            if arg.annotation:
                                param_info["type"] = ast.unparse(arg.annotation)
                            
                            analysis["parameters"].append(param_info)
                        
                        # Extract return type
                        if node.returns:
                            analysis["return_type"] = ast.unparse(node.returns)
                        
                        break
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)
            
            # Extract examples from docstring
            if analysis["docstring"]:
                examples = self._extract_examples_from_docstring(analysis["docstring"])
                analysis["examples"] = examples
            
            # Identify potential edge cases
            analysis["edge_cases"] = self._identify_edge_cases(code, analysis)
            
        except Exception as e:
            logger.warning(f"Code analysis failed: {e}")
        
        return analysis
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        
        # Look for code blocks in docstring
        code_block_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_block_pattern, docstring, re.DOTALL)
        examples.extend(matches)
        
        # Look for example sections
        example_pattern = r'Example[s]?:\s*\n(.*?)(?:\n\n|\Z)'
        matches = re.findall(example_pattern, docstring, re.DOTALL | re.IGNORECASE)
        examples.extend(matches)
        
        return examples
    
    def _identify_edge_cases(self, code: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential edge cases from code analysis."""
        edge_cases = []
        
        # Common edge case patterns
        if "if not" in code:
            edge_cases.append("Empty or None input handling")
        
        if "try:" in code and "except" in code:
            edge_cases.append("Exception handling for invalid inputs")
        
        if "len(" in code:
            edge_cases.append("Empty collection handling")
        
        if "isinstance" in code:
            edge_cases.append("Type validation and conversion")
        
        # Parameter-specific edge cases
        for param in analysis["parameters"]:
            param_type = param.get("type", "").lower()
            
            if "str" in param_type:
                edge_cases.append(f"Empty string handling for {param['name']}")
            elif "int" in param_type or "float" in param_type:
                edge_cases.append(f"Negative/zero values for {param['name']}")
            elif "list" in param_type or "dict" in param_type:
                edge_cases.append(f"Empty collection handling for {param['name']}")
        
        return list(set(edge_cases))  # Remove duplicates
    
    def _create_documentation_content(self, tool_name: str, metadata: ToolMetadata, analysis: Dict[str, Any]) -> str:
        """Create comprehensive documentation content."""
        doc_content = f"""# {tool_name}

## Overview

{metadata.description}

**Status:** {metadata.status.value.title()}  
**Version:** {metadata.version}  
**Category:** {metadata.category}  
**Author:** {metadata.author}  
**Created:** {metadata.created_at.strftime('%Y-%m-%d')}  
**Last Updated:** {metadata.updated_at.strftime('%Y-%m-%d')}  

"""
        
        # Add deprecation warning if applicable
        if metadata.status == ToolStatus.DEPRECATED:
            doc_content += f"""## ⚠️ Deprecation Notice

This tool is deprecated and will be removed on {metadata.removal_date.strftime('%Y-%m-%d') if metadata.removal_date else 'TBD'}.

**Reason:** {metadata.deprecation_reason}  
**Replacement:** {metadata.replacement_tool if metadata.replacement_tool else 'None specified'}  

"""
        
        # Function signature
        if analysis["function_name"]:
            doc_content += f"""## Function Signature

```python
{analysis["function_name"]}(
"""
            for param in analysis["parameters"]:
                default_str = f" = {param['default']}" if param['default'] else ""
                doc_content += f"    {param['name']}: {param['type']}{default_str},\n"
            
            doc_content += f") -> {analysis['return_type'] or 'Any'}\n```\n\n"
        
        # Description from docstring
        if analysis["docstring"]:
            doc_content += f"""## Description

{analysis["docstring"]}

"""
        
        # Parameters
        if analysis["parameters"]:
            doc_content += """## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
"""
            for param in analysis["parameters"]:
                default_val = param['default'] if param['default'] else 'Required'
                # Extract parameter description from docstring if available
                param_desc = self._extract_parameter_description(analysis["docstring"], param['name'])
                doc_content += f"| `{param['name']}` | `{param['type']}` | {param_desc} | {default_val} |\n"
            
            doc_content += "\n"
        
        # Return value
        if analysis["return_type"]:
            return_desc = self._extract_return_description(analysis["docstring"])
            doc_content += f"""## Return Value

**Type:** `{analysis['return_type']}`

{return_desc}

"""
        
        # Dependencies
        if metadata.dependencies:
            doc_content += f"""## Dependencies

{', '.join(f'`{dep}`' for dep in metadata.dependencies)}

"""
        
        # Usage examples
        if analysis["examples"]:
            doc_content += """## Usage Examples

"""
            for i, example in enumerate(analysis["examples"], 1):
                doc_content += f"""### Example {i}

```python
{example.strip()}
```

"""
        else:
            # Generate basic example
            doc_content += f"""## Usage Examples

### Basic Usage

```python
from tools.{tool_name} import {tool_name}

# Basic usage
result = {tool_name}.invoke({{"param": "value"}})
print(result)
```

"""
        
        # Edge cases
        if analysis["edge_cases"]:
            doc_content += """## Edge Cases and Considerations

"""
            for edge_case in analysis["edge_cases"]:
                doc_content += f"- {edge_case}\n"
            
            doc_content += "\n"
        
        # Error handling
        doc_content += """## Error Handling

This tool includes error handling for common failure scenarios:

- **Invalid Input:** Returns appropriate error messages for invalid parameters
- **Missing Dependencies:** Graceful handling when required dependencies are unavailable
- **Runtime Errors:** Comprehensive exception handling with descriptive error messages

"""
        
        # Performance notes
        doc_content += """## Performance Notes

- **Execution Time:** Typically completes within seconds
- **Memory Usage:** Minimal memory footprint
- **Concurrency:** Safe for concurrent execution

"""
        
        # Version history
        doc_content += f"""## Version History

| Version | Date | Changes |
|---------|------|---------|
| {metadata.version} | {metadata.updated_at.strftime('%Y-%m-%d')} | Current version |

"""
        
        # Usage statistics
        if metadata.usage_count > 0:
            doc_content += f"""## Usage Statistics

- **Total Uses:** {metadata.usage_count}
- **Last Used:** {metadata.last_used.strftime('%Y-%m-%d %H:%M') if metadata.last_used else 'Never'}

"""
        
        # Footer
        doc_content += f"""---

*Documentation auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return doc_content
    
    def _extract_parameter_description(self, docstring: str, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring:
            return "No description available"
        
        # Look for Args section
        args_pattern = r'Args:\s*\n(.*?)(?:\n\n|\nReturns:|\nRaises:|\Z)'
        args_match = re.search(args_pattern, docstring, re.DOTALL)
        
        if args_match:
            args_section = args_match.group(1)
            # Look for specific parameter
            param_pattern = rf'{param_name}\s*\([^)]*\):\s*([^\n]+)'
            param_match = re.search(param_pattern, args_section)
            if param_match:
                return param_match.group(1).strip()
        
        return "No description available"
    
    def _extract_return_description(self, docstring: str) -> str:
        """Extract return value description from docstring."""
        if not docstring:
            return "No description available"
        
        # Look for Returns section
        returns_pattern = r'Returns:\s*\n(.*?)(?:\n\n|\nRaises:|\Z)'
        returns_match = re.search(returns_pattern, docstring, re.DOTALL)
        
        if returns_match:
            return returns_match.group(1).strip()
        
        return "No description available"


class ToolLifecycleManager:
    """Tool deprecation and lifecycle management."""
    
    def __init__(self, metadata_file: str = "tools_metadata.json"):
        self.metadata_file = Path(metadata_file)
        self.tools_metadata: Dict[str, ToolMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load tool metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for tool_name, tool_data in data.items():
                    # Convert datetime strings back to datetime objects
                    tool_data["created_at"] = datetime.fromisoformat(tool_data["created_at"])
                    tool_data["updated_at"] = datetime.fromisoformat(tool_data["updated_at"])
                    if tool_data.get("last_used"):
                        tool_data["last_used"] = datetime.fromisoformat(tool_data["last_used"])
                    if tool_data.get("deprecation_date"):
                        tool_data["deprecation_date"] = datetime.fromisoformat(tool_data["deprecation_date"])
                    if tool_data.get("removal_date"):
                        tool_data["removal_date"] = datetime.fromisoformat(tool_data["removal_date"])
                    
                    # Convert status string to enum
                    tool_data["status"] = ToolStatus(tool_data["status"])
                    
                    self.tools_metadata[tool_name] = ToolMetadata(**tool_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load tool metadata: {e}")
    
    def _save_metadata(self):
        """Save tool metadata to file."""
        try:
            data = {}
            for tool_name, metadata in self.tools_metadata.items():
                data[tool_name] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save tool metadata: {e}")
    
    def register_tool(self, tool_name: str, author: str = "ToolMaker", 
                     description: str = "", category: str = "general") -> ToolMetadata:
        """Register a new tool in the lifecycle system."""
        metadata = ToolMetadata(
            name=tool_name,
            version="1.0.0",
            status=ToolStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=author,
            description=description,
            category=category
        )
        
        self.tools_metadata[tool_name] = metadata
        self._save_metadata()
        
        logger.info(f"Registered tool: {tool_name}")
        return metadata
    
    def deprecate_tool(self, tool_name: str, reason: str, replacement_tool: str = "", 
                      deprecation_period_days: int = 90) -> bool:
        """Mark a tool as deprecated."""
        if tool_name not in self.tools_metadata:
            logger.error(f"Tool not found: {tool_name}")
            return False
        
        metadata = self.tools_metadata[tool_name]
        metadata.status = ToolStatus.DEPRECATED
        metadata.deprecation_date = datetime.now()
        metadata.removal_date = datetime.now() + timedelta(days=deprecation_period_days)
        metadata.deprecation_reason = reason
        metadata.replacement_tool = replacement_tool
        metadata.updated_at = datetime.now()
        
        self._save_metadata()
        
        logger.info(f"Deprecated tool: {tool_name} (removal: {metadata.removal_date})")
        return True
    
    def update_usage(self, tool_name: str):
        """Update tool usage statistics."""
        if tool_name in self.tools_metadata:
            metadata = self.tools_metadata[tool_name]
            metadata.usage_count += 1
            metadata.last_used = datetime.now()
            self._save_metadata()
    
    def get_deprecated_tools(self) -> List[ToolMetadata]:
        """Get list of deprecated tools."""
        return [metadata for metadata in self.tools_metadata.values() 
                if metadata.status == ToolStatus.DEPRECATED]
    
    def get_tools_for_removal(self) -> List[ToolMetadata]:
        """Get tools that are ready for removal."""
        now = datetime.now()
        return [metadata for metadata in self.tools_metadata.values()
                if (metadata.status == ToolStatus.DEPRECATED and 
                    metadata.removal_date and metadata.removal_date <= now)]
    
    def get_unused_tools(self, days_threshold: int = 30) -> List[ToolMetadata]:
        """Get tools that haven't been used recently."""
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        return [metadata for metadata in self.tools_metadata.values()
                if (metadata.last_used is None or metadata.last_used < threshold_date)]
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the system."""
        if tool_name not in self.tools_metadata:
            return False
        
        try:
            # Remove tool file
            tool_file = Path(f"tools/{tool_name}.py")
            if tool_file.exists():
                tool_file.unlink()
            
            # Remove test file
            test_file = Path(f"tests/tools/test_{tool_name}.py")
            if test_file.exists():
                test_file.unlink()
            
            # Remove documentation
            doc_file = Path(f"docs/tools/{tool_name}.md")
            if doc_file.exists():
                doc_file.unlink()
            
            # Remove from metadata
            del self.tools_metadata[tool_name]
            self._save_metadata()
            
            logger.info(f"Removed tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove tool {tool_name}: {e}")
            return False


# Initialize enhanced components
tool_validator = ToolValidator()
doc_generator = ToolDocumentationGenerator()
lifecycle_manager = ToolLifecycleManager()

# Enhanced system prompt
enhanced_system_prompt = """You are Enhanced Tool Maker, an advanced ReAct agent that develops LangChain tools with comprehensive validation, documentation, and lifecycle management.

ENHANCED CAPABILITIES:
1. **Tool Validation**: Sandboxed testing for functionality and safety
2. **Documentation Generation**: Auto-generated comprehensive documentation
3. **Lifecycle Management**: Complete deprecation and removal process

YOUR ENHANCED WORKFLOW:
1. **Analyze Request** - Understand requirements and check for existing tools
2. **Design & Plan** - Plan tool architecture with validation in mind
3. **Write Code & Test** - Create tool with comprehensive error handling
4. **Sandbox Validation** - Test in isolated environment for safety and functionality
5. **Generate Documentation** - Auto-create comprehensive usage documentation
6. **Register Tool** - Add to lifecycle management system
7. **Format & Lint** - Ensure code quality standards
8. **Final Validation** - Complete testing and validation
9. **Deploy** - Deploy with monitoring and deprecation tracking

ENHANCED REQUIREMENTS:
- **ALWAYS** validate tools in sandbox before deployment
- **ALWAYS** generate comprehensive documentation
- **ALWAYS** register tools in lifecycle management system
- **NEVER** deploy tools that fail validation
- **MONITOR** tool usage and lifecycle status

ENHANCED TOOL FORMAT:
```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ToolInput(BaseModel):
    \"\"\"Input schema for the tool.\"\"\"
    param1: str = Field(description="Description of parameter")
    param2: Optional[int] = Field(default=None, description="Optional parameter")

@tool(args_schema=ToolInput)
def tool_name(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    \"\"\"
    Enhanced tool with comprehensive documentation.
    
    Args:
        param1 (str): Description of parameter
        param2 (Optional[int]): Optional parameter description
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
        
    Raises:
        ValueError: When input validation fails
        
    Examples:
        >>> result = tool_name.invoke({"param1": "test"})
        >>> print(result["status"])
        success
    \"\"\"
    try:
        # Input validation
        if not param1:
            raise ValueError("param1 cannot be empty")
        
        logger.info(f"Executing tool_name with param1: {param1}")
        
        # Tool logic here
        result = f"Processed: {param1}"
        
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully processed: {param1}"
        }
        
    except Exception as e:
        logger.error(f"Error in tool_name: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Tool execution failed: {str(e)}"
        }
```

VALIDATION REQUIREMENTS:
- Security validation (no dangerous operations)
- Functionality validation (proper imports and structure)
- Performance validation (reasonable size and complexity)
- Documentation validation (comprehensive docs)

Use enhanced tools for validation, documentation, and lifecycle management throughout the process.
"""

# Load enhanced tools
tools = utils.all_tool_functions()


def enhanced_tool_maker(task: str) -> Dict[str, Any]:
    """
    Enhanced Tool Maker with validation, documentation, and lifecycle management.
    
    Args:
        task (str): The description of the tool to be created.
        
    Returns:
        Dict[str, Any]: Enhanced result with validation and documentation information
    """
    logger.info(f"Enhanced Tool Maker invoked for task: {task[:100]}...")
    
    try:
        # Create enhanced workflow
        enhanced_workflow = _create_enhanced_workflow()
        
        # Execute enhanced workflow
        final_state = enhanced_workflow.invoke({
            "messages": [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=task)
            ]
        })
        
        # Extract and analyze results
        messages_history = final_state.get("messages", []) if final_state else []
        status, tool_name, final_message = _analyze_enhanced_result(messages_history)
        
        # Enhanced post-processing
        enhanced_result = {
            "status": status,
            "result": tool_name,
            "message": final_message,
            "validation_report": None,
            "documentation_path": None,
            "lifecycle_status": None
        }
        
        if status == "success" and tool_name:
            # Validate tool
            tool_path = f"tools/{tool_name}.py"
            if os.path.exists(tool_path):
                validation_report = tool_validator.validate_tool(tool_path, tool_name)
                enhanced_result["validation_report"] = validation_report.to_dict()
                
                if validation_report.overall_result == ValidationResult.FAILED:
                    enhanced_result["status"] = "failure"
                    enhanced_result["message"] = f"Tool validation failed: {', '.join(validation_report.errors)}"
                    return enhanced_result
                
                # Register in lifecycle management
                metadata = lifecycle_manager.register_tool(
                    tool_name=tool_name,
                    description=task[:200],
                    category=_determine_tool_category(task)
                )
                enhanced_result["lifecycle_status"] = metadata.to_dict()
                
                # Generate documentation
                doc_path = doc_generator.generate_documentation(tool_path, tool_name, metadata)
                enhanced_result["documentation_path"] = doc_path
                
                # Update manifest
                if utils.add_manifest_entry("tool", {
                    "name": tool_name,
                    "module_path": f"tools/{tool_name}.py",
                    "function_name": tool_name,
                    "description": task[:200],
                    "status": "active",
                    "validation_score": validation_report.overall_result.value,
                    "documentation": doc_path
                }):
                    enhanced_result["message"] = f"Successfully created, validated, documented, and registered tool: '{tool_name}'"
                else:
                    enhanced_result["status"] = "failure"
                    enhanced_result["message"] = "Tool created but failed to register in manifest"
        
        return enhanced_result
        
    except Exception as e:
        error_msg = f"Enhanced Tool Maker execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "validation_report": None,
            "documentation_path": None,
            "lifecycle_status": None
        }


def _create_enhanced_workflow():
    """Create the enhanced LangGraph workflow."""
    def enhanced_reasoning(state: MessagesState) -> dict:
        """Enhanced reasoning with validation awareness."""
        print("\nenhanced tool_maker is thinking...")
        
        # Get memory context
        current_messages = state["messages"]
        task_description = ""
        
        for msg in reversed(current_messages):
            if isinstance(msg, HumanMessage):
                task_description = str(msg.content)
                break
        
        memory_context, template_context = get_memory_context(task_description)
        
        # Prepare enhanced messages
        messages_for_llm = list(current_messages)
        combined_context = template_context + memory_context
        
        if combined_context and messages_for_llm:
            if isinstance(messages_for_llm[0], SystemMessage):
                original_content = messages_for_llm[0].content
                messages_for_llm[0] = SystemMessage(content=f"{combined_context}{original_content}")
        
        try:
            # Use Google Gemini for tool calling from hybrid configuration
            tool_model = config.get_model_for_tools()
            if tool_model is None:
                # Fallback to default model if hybrid setup fails
                tool_model = config.default_langchain_model
                logger.warning("Using fallback model for tools - may not support function calling")
            
            tooled_up_model = tool_model.bind_tools(tools)
            response = tooled_up_model.invoke(messages_for_llm)
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Enhanced reasoning error: {e}", exc_info=True)
            return {"messages": [AIMessage(content=f"Enhanced Tool Maker error: {str(e)}")]}
    
    def enhanced_check_for_tool_calls(state: MessagesState) -> Literal["tools", "END"]:
        """Enhanced tool call checking."""
        messages = state["messages"]
        if not messages:
            return "END"
        
        last_message = messages[-1]
        
        if (isinstance(last_message, AIMessage) and 
            hasattr(last_message, "tool_calls") and last_message.tool_calls):
            return "tools"
        
        return "END"
    
    # Create enhanced workflow
    workflow = StateGraph(MessagesState)
    workflow.add_node("reasoning", enhanced_reasoning)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("reasoning")
    workflow.add_conditional_edges("reasoning", enhanced_check_for_tool_calls)
    workflow.add_edge("tools", "reasoning")
    
    return workflow.compile()


def _analyze_enhanced_result(messages_history: list) -> Tuple[str, Optional[str], str]:
    """Analyze enhanced tool creation result."""
    status = "failure"
    tool_name = None
    final_message = "Enhanced tool creation failed"
    
    if not messages_history:
        return status, tool_name, final_message
    
    # Get final message
    last_message = messages_history[-1]
    if hasattr(last_message, "content"):
        final_message = str(last_message.content)
    
    # Look for success indicators
    success_patterns = [
        r"successfully created.*?tool.*?['\"]([^'\"]+)['\"]",
        r"tool.*?['\"]([^'\"]+)['\"].*?successfully",
        r"created and tested.*?['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in success_patterns:
        match = re.search(pattern, final_message.lower())
        if match:
            potential_tool_name = match.group(1)
            if potential_tool_name:
                status = "success"
                tool_name = potential_tool_name
                break
    
    return status, tool_name, final_message


def _determine_tool_category(task_description: str) -> str:
    """Determine tool category based on task description."""
    task_lower = task_description.lower()
    
    if any(keyword in task_lower for keyword in ["file", "read", "write", "directory"]):
        return "file_management"
    elif any(keyword in task_lower for keyword in ["web", "http", "api", "request"]):
        return "web_interaction"
    elif any(keyword in task_lower for keyword in ["data", "process", "analyze", "transform"]):
        return "data_processing"
    elif any(keyword in task_lower for keyword in ["text", "string", "format", "parse"]):
        return "text_processing"
    elif any(keyword in task_lower for keyword in ["math", "calculate", "compute"]):
        return "computation"
    elif any(keyword in task_lower for keyword in ["system", "command", "shell"]):
        return "system_interaction"
    else:
        return "general"


def get_memory_context(task_description: str) -> Tuple[str, str]:
    """Enhanced memory context retrieval."""
    memory_context = ""
    template_context = ""
    
    if not MEMORY_AVAILABLE or not memory_manager_instance:
        return memory_context, template_context
    
    try:
        # Retrieve relevant memories
        relevant_memories = []
        if hasattr(memory_manager_instance, "retrieve_memories"):
            relevant_memories = memory_manager_instance.retrieve_memories(limit=5)
        
        if relevant_memories:
            memory_context = "Relevant context from past interactions:\n"
            memory_context += "\n".join([f"- {mem}" for mem in relevant_memories])
            memory_context += "\n\n---\n\n"
        
        # Get tool templates
        tool_templates = []
        if hasattr(memory_manager_instance, "retrieve_memories"):
            tool_templates = memory_manager_instance.retrieve_memories(memory_type="tool_template", limit=1)
        
        if tool_templates:
            template_context = f"Enhanced template found:\n```python\n{tool_templates[0]}\n```\n\n---\n\n"
    
    except Exception as e:
        logger.warning(f"Failed to retrieve enhanced memories: {e}")
    
    return memory_context, template_context


# Export enhanced function
__all__ = ["enhanced_tool_maker", "ToolValidator", "ToolDocumentationGenerator", "ToolLifecycleManager"]