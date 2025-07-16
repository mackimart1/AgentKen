from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import os
import ast
import re
from datetime import datetime
from pathlib import Path


class GenerateDocsInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to document")
    tool_path: str = Field(description="Path to the tool file")
    output_format: str = Field(default="markdown", description="Output format: markdown, html, json")
    include_examples: bool = Field(default=True, description="Include usage examples")


class UpdateDocsInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to update documentation for")
    additional_info: Dict[str, Any] = Field(default={}, description="Additional information to include")


class ValidateDocsInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to validate documentation for")
    check_completeness: bool = Field(default=True, description="Check documentation completeness")


# Global documentation storage
_docs_cache: Dict[str, Dict[str, Any]] = {}
_docs_dir = Path("docs/tools")
_docs_dir.mkdir(parents=True, exist_ok=True)


def _analyze_tool_structure(tool_path: str) -> Dict[str, Any]:
    """Analyze tool file structure for documentation generation."""
    analysis = {
        "function_name": "",
        "docstring": "",
        "parameters": [],
        "return_type": "",
        "imports": [],
        "decorators": [],
        "examples": [],
        "error_handling": False,
        "input_schema": None,
        "complexity_score": 0
    }
    
    try:
        with open(tool_path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Find tool function and analyze
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for @tool decorator
                has_tool_decorator = any(
                    (isinstance(d, ast.Name) and d.id == 'tool') or
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'tool')
                    for d in node.decorator_list
                )
                
                if has_tool_decorator:
                    analysis["function_name"] = node.name
                    
                    # Extract docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant)):
                        analysis["docstring"] = node.body[0].value.value
                    
                    # Extract parameters
                    for arg in node.args.args:
                        param_info = {
                            "name": arg.arg,
                            "type": "Any",
                            "default": None,
                            "description": ""
                        }
                        
                        # Get type annotation
                        if arg.annotation:
                            param_info["type"] = ast.unparse(arg.annotation)
                        
                        analysis["parameters"].append(param_info)
                    
                    # Extract return type
                    if node.returns:
                        analysis["return_type"] = ast.unparse(node.returns)
                    
                    # Check for error handling
                    for child in ast.walk(node):
                        if isinstance(child, ast.Try):
                            analysis["error_handling"] = True
                            break
                    
                    # Calculate complexity
                    analysis["complexity_score"] = len(list(ast.walk(node)))
                    
                    break
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imports"].append(node.module)
        
        # Look for input schema class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a Pydantic model for input schema
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseModel":
                        analysis["input_schema"] = node.name
                        break
        
        # Extract examples from docstring
        if analysis["docstring"]:
            examples = _extract_examples_from_docstring(analysis["docstring"])
            analysis["examples"] = examples
        
    except Exception as e:
        print(f"Warning: Failed to analyze tool structure: {e}")
    
    return analysis


def _extract_examples_from_docstring(docstring: str) -> List[Dict[str, str]]:
    """Extract examples from docstring."""
    examples = []
    
    # Look for Examples section
    example_pattern = r'Examples?:\s*\n(.*?)(?:\n\n|\n[A-Z]|\Z)'
    match = re.search(example_pattern, docstring, re.DOTALL | re.IGNORECASE)
    
    if match:
        examples_text = match.group(1)
        
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', examples_text, re.DOTALL)
        for i, code in enumerate(code_blocks):
            examples.append({
                "title": f"Example {i + 1}",
                "code": code.strip(),
                "description": "Code example from docstring"
            })
        
        # Look for >>> style examples
        doctest_examples = re.findall(r'>>> (.*?)(?:\n|$)', examples_text)
        if doctest_examples:
            examples.append({
                "title": "Interactive Example",
                "code": "\n".join(doctest_examples),
                "description": "Interactive usage example"
            })
    
    return examples


def _extract_parameter_info(docstring: str, param_name: str) -> Dict[str, str]:
    """Extract parameter information from docstring."""
    info = {"description": "", "type": "", "default": ""}
    
    if not docstring:
        return info
    
    # Look for Args section
    args_pattern = r'Args:\s*\n(.*?)(?:\n\n|\nReturns:|\nRaises:|\Z)'
    args_match = re.search(args_pattern, docstring, re.DOTALL)
    
    if args_match:
        args_section = args_match.group(1)
        
        # Look for parameter description
        param_pattern = rf'{param_name}\s*\(([^)]*)\):\s*([^\n]+)'
        param_match = re.search(param_pattern, args_section)
        
        if param_match:
            info["type"] = param_match.group(1).strip()
            info["description"] = param_match.group(2).strip()
        else:
            # Try simpler pattern
            simple_pattern = rf'{param_name}:\s*([^\n]+)'
            simple_match = re.search(simple_pattern, args_section)
            if simple_match:
                info["description"] = simple_match.group(1).strip()
    
    return info


def _generate_usage_examples(tool_name: str, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate usage examples for the tool."""
    examples = []
    
    # Basic usage example
    basic_example = {
        "title": "Basic Usage",
        "description": "Simple example of using the tool",
        "code": f"""from tools.{tool_name} import {tool_name}

# Basic usage
result = {tool_name}.invoke({{"param": "value"}})
print(result)"""
    }
    examples.append(basic_example)
    
    # Parameter-specific examples
    if analysis["parameters"]:
        param_example = {
            "title": "With Parameters",
            "description": "Example using specific parameters",
            "code": f"""from tools.{tool_name} import {tool_name}

# Using specific parameters
result = {tool_name}.invoke({{
"""
        }
        
        for param in analysis["parameters"][:3]:  # Show first 3 parameters
            if param["name"] != "self":
                param_example["code"] += f'    "{param["name"]}": "example_value",\n'
        
        param_example["code"] += "})\nprint(result)"
        examples.append(param_example)
    
    # Error handling example
    if analysis["error_handling"]:
        error_example = {
            "title": "Error Handling",
            "description": "Example with error handling",
            "code": f"""from tools.{tool_name} import {tool_name}

try:
    result = {tool_name}.invoke({{"param": "value"}})
    if result.get("status") == "success":
        print("Success:", result["result"])
    else:
        print("Error:", result["message"])
except Exception as e:
    print("Exception:", str(e))"""
        }
        examples.append(error_example)
    
    return examples


def _identify_edge_cases(analysis: Dict[str, Any]) -> List[str]:
    """Identify potential edge cases for the tool."""
    edge_cases = []
    
    # Parameter-based edge cases
    for param in analysis["parameters"]:
        param_type = param.get("type", "").lower()
        param_name = param["name"]
        
        if "str" in param_type:
            edge_cases.append(f"Empty string for {param_name}")
            edge_cases.append(f"Very long string for {param_name}")
            edge_cases.append(f"String with special characters for {param_name}")
        
        elif "int" in param_type or "float" in param_type:
            edge_cases.append(f"Zero value for {param_name}")
            edge_cases.append(f"Negative value for {param_name}")
            edge_cases.append(f"Very large value for {param_name}")
        
        elif "list" in param_type:
            edge_cases.append(f"Empty list for {param_name}")
            edge_cases.append(f"List with None values for {param_name}")
        
        elif "dict" in param_type:
            edge_cases.append(f"Empty dictionary for {param_name}")
            edge_cases.append(f"Dictionary with missing keys for {param_name}")
        
        elif "optional" in param_type.lower() or "none" in param_type.lower():
            edge_cases.append(f"None value for {param_name}")
    
    # General edge cases
    edge_cases.extend([
        "Network connectivity issues (if applicable)",
        "File system permissions (if applicable)",
        "Memory limitations with large inputs",
        "Concurrent execution scenarios"
    ])
    
    return edge_cases


@tool(args_schema=GenerateDocsInput)
def generate_tool_documentation(
    tool_name: str,
    tool_path: str,
    output_format: str = "markdown",
    include_examples: bool = True
) -> str:
    """
    Generate comprehensive documentation for a tool.
    
    Args:
        tool_name: Name of the tool to document
        tool_path: Path to the tool file
        output_format: Output format (markdown, html, json)
        include_examples: Include usage examples
    
    Returns:
        JSON string with documentation generation results
    """
    try:
        if not os.path.exists(tool_path):
            return json.dumps({
                "status": "failure",
                "message": f"Tool file not found: {tool_path}"
            })
        
        # Analyze tool structure
        analysis = _analyze_tool_structure(tool_path)
        
        if not analysis["function_name"]:
            return json.dumps({
                "status": "failure",
                "message": "No tool function found in file"
            })
        
        # Generate documentation content
        if output_format == "markdown":
            doc_content = _generate_markdown_documentation(tool_name, analysis, include_examples)
            file_extension = ".md"
        elif output_format == "html":
            doc_content = _generate_html_documentation(tool_name, analysis, include_examples)
            file_extension = ".html"
        elif output_format == "json":
            doc_content = _generate_json_documentation(tool_name, analysis, include_examples)
            file_extension = ".json"
        else:
            return json.dumps({
                "status": "failure",
                "message": f"Unsupported output format: {output_format}"
            })
        
        # Save documentation
        doc_file = _docs_dir / f"{tool_name}{file_extension}"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        # Cache documentation info
        _docs_cache[tool_name] = {
            "file_path": str(doc_file),
            "format": output_format,
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis
        }
        
        return json.dumps({
            "status": "success",
            "documentation_path": str(doc_file),
            "format": output_format,
            "analysis_summary": {
                "function_name": analysis["function_name"],
                "parameter_count": len(analysis["parameters"]),
                "has_docstring": bool(analysis["docstring"]),
                "has_examples": len(analysis["examples"]) > 0,
                "complexity_score": analysis["complexity_score"]
            },
            "message": f"Documentation generated successfully for {tool_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Documentation generation failed: {str(e)}"
        })


def _generate_markdown_documentation(tool_name: str, analysis: Dict[str, Any], include_examples: bool) -> str:
    """Generate Markdown documentation."""
    doc = f"""# {tool_name}

## Overview

{analysis.get('docstring', 'No description available').split('.')[0] if analysis.get('docstring') else 'Tool for performing specific operations.'}

**Function:** `{analysis['function_name']}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

"""
    
    # Function signature
    if analysis["function_name"]:
        doc += f"""## Function Signature

```python
{analysis['function_name']}(
"""
        for param in analysis["parameters"]:
            if param["name"] != "self":
                doc += f"    {param['name']}: {param['type']},\n"
        
        doc += f") -> {analysis['return_type'] or 'Any'}\n```\n\n"
    
    # Description
    if analysis["docstring"]:
        doc += f"""## Description

{analysis['docstring']}

"""
    
    # Parameters
    if analysis["parameters"]:
        doc += """## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
"""
        for param in analysis["parameters"]:
            if param["name"] != "self":
                param_info = _extract_parameter_info(analysis["docstring"], param["name"])
                description = param_info["description"] or "No description available"
                required = "Yes" if "optional" not in param["type"].lower() else "No"
                doc += f"| `{param['name']}` | `{param['type']}` | {description} | {required} |\n"
        
        doc += "\n"
    
    # Return value
    if analysis["return_type"]:
        doc += f"""## Return Value

**Type:** `{analysis['return_type']}`

Returns the result of the tool operation.

"""
    
    # Usage examples
    if include_examples:
        doc += """## Usage Examples

"""
        
        # Use existing examples or generate new ones
        examples = analysis["examples"] if analysis["examples"] else _generate_usage_examples(tool_name, analysis)
        
        for example in examples:
            doc += f"""### {example['title']}

{example.get('description', '')}

```python
{example['code']}
```

"""
    
    # Edge cases
    edge_cases = _identify_edge_cases(analysis)
    if edge_cases:
        doc += """## Edge Cases and Considerations

"""
        for edge_case in edge_cases[:5]:  # Show first 5
            doc += f"- {edge_case}\n"
        
        doc += "\n"
    
    # Error handling
    if analysis["error_handling"]:
        doc += """## Error Handling

This tool includes comprehensive error handling:

- Input validation with descriptive error messages
- Exception handling for runtime errors
- Graceful degradation for edge cases

"""
    
    # Technical details
    doc += f"""## Technical Details

- **Complexity Score:** {analysis['complexity_score']}
- **Dependencies:** {', '.join(analysis['imports'][:5]) if analysis['imports'] else 'None'}
- **Input Schema:** {analysis['input_schema'] or 'Dynamic'}

"""
    
    # Footer
    doc += f"""---

*Documentation auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return doc


def _generate_html_documentation(tool_name: str, analysis: Dict[str, Any], include_examples: bool) -> str:
    """Generate HTML documentation."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tool_name} - Tool Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .code {{ background-color: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{tool_name}</h1>
        <p><strong>Function:</strong> {analysis['function_name']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    # Description
    if analysis["docstring"]:
        html += f"""
    <div class="section">
        <h2>Description</h2>
        <p>{analysis['docstring']}</p>
    </div>
"""
    
    # Parameters
    if analysis["parameters"]:
        html += """
    <div class="section">
        <h2>Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
"""
        for param in analysis["parameters"]:
            if param["name"] != "self":
                param_info = _extract_parameter_info(analysis["docstring"], param["name"])
                description = param_info["description"] or "No description available"
                html += f"            <tr><td>{param['name']}</td><td>{param['type']}</td><td>{description}</td></tr>\n"
        
        html += """        </table>
    </div>
"""
    
    # Examples
    if include_examples:
        examples = analysis["examples"] if analysis["examples"] else _generate_usage_examples(tool_name, analysis)
        if examples:
            html += """
    <div class="section">
        <h2>Usage Examples</h2>
"""
            for example in examples:
                html += f"""
        <h3>{example['title']}</h3>
        <p>{example.get('description', '')}</p>
        <div class="code">{example['code']}</div>
"""
            html += "    </div>\n"
    
    html += """
</body>
</html>
"""
    
    return html


def _generate_json_documentation(tool_name: str, analysis: Dict[str, Any], include_examples: bool) -> str:
    """Generate JSON documentation."""
    doc_data = {
        "tool_name": tool_name,
        "function_name": analysis["function_name"],
        "generated_at": datetime.now().isoformat(),
        "description": analysis.get("docstring", ""),
        "parameters": [],
        "return_type": analysis["return_type"],
        "examples": [],
        "edge_cases": _identify_edge_cases(analysis),
        "technical_details": {
            "complexity_score": analysis["complexity_score"],
            "dependencies": analysis["imports"],
            "input_schema": analysis["input_schema"],
            "has_error_handling": analysis["error_handling"]
        }
    }
    
    # Process parameters
    for param in analysis["parameters"]:
        if param["name"] != "self":
            param_info = _extract_parameter_info(analysis["docstring"], param["name"])
            doc_data["parameters"].append({
                "name": param["name"],
                "type": param["type"],
                "description": param_info["description"] or "No description available",
                "required": "optional" not in param["type"].lower()
            })
    
    # Add examples
    if include_examples:
        examples = analysis["examples"] if analysis["examples"] else _generate_usage_examples(tool_name, analysis)
        doc_data["examples"] = examples
    
    return json.dumps(doc_data, indent=2)


@tool(args_schema=UpdateDocsInput)
def update_tool_documentation(tool_name: str, additional_info: Dict[str, Any] = None) -> str:
    """
    Update existing tool documentation with additional information.
    
    Args:
        tool_name: Name of the tool to update documentation for
        additional_info: Additional information to include
    
    Returns:
        JSON string with update results
    """
    try:
        if additional_info is None:
            additional_info = {}
        
        if tool_name not in _docs_cache:
            return json.dumps({
                "status": "failure",
                "message": f"No documentation found for tool: {tool_name}"
            })
        
        doc_info = _docs_cache[tool_name]
        doc_file = Path(doc_info["file_path"])
        
        if not doc_file.exists():
            return json.dumps({
                "status": "failure",
                "message": f"Documentation file not found: {doc_file}"
            })
        
        # Read existing documentation
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add additional information
        if doc_info["format"] == "markdown":
            # Add additional info section
            additional_section = "\n## Additional Information\n\n"
            for key, value in additional_info.items():
                additional_section += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
            
            # Insert before footer
            if "---" in content:
                content = content.replace("---", additional_section + "---")
            else:
                content += additional_section
        
        elif doc_info["format"] == "json":
            # Parse JSON and add additional info
            doc_data = json.loads(content)
            doc_data["additional_info"] = additional_info
            doc_data["last_updated"] = datetime.now().isoformat()
            content = json.dumps(doc_data, indent=2)
        
        # Save updated documentation
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update cache
        _docs_cache[tool_name]["last_updated"] = datetime.now().isoformat()
        
        return json.dumps({
            "status": "success",
            "updated_file": str(doc_file),
            "additional_info": additional_info,
            "message": f"Documentation updated for {tool_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Documentation update failed: {str(e)}"
        })


@tool(args_schema=ValidateDocsInput)
def validate_tool_documentation(tool_name: str, check_completeness: bool = True) -> str:
    """
    Validate tool documentation for completeness and quality.
    
    Args:
        tool_name: Name of the tool to validate documentation for
        check_completeness: Check documentation completeness
    
    Returns:
        JSON string with validation results
    """
    try:
        if tool_name not in _docs_cache:
            return json.dumps({
                "status": "failure",
                "message": f"No documentation found for tool: {tool_name}"
            })
        
        doc_info = _docs_cache[tool_name]
        doc_file = Path(doc_info["file_path"])
        
        if not doc_file.exists():
            return json.dumps({
                "status": "failure",
                "message": f"Documentation file not found: {doc_file}"
            })
        
        validation_result = {
            "tool_name": tool_name,
            "file_path": str(doc_file),
            "format": doc_info["format"],
            "completeness_score": 0,
            "quality_score": 0,
            "issues": [],
            "recommendations": []
        }
        
        # Read documentation
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if check_completeness:
            # Check completeness
            completeness_checks = {
                "has_description": bool(re.search(r'description|overview', content, re.IGNORECASE)),
                "has_parameters": bool(re.search(r'parameter|argument', content, re.IGNORECASE)),
                "has_examples": bool(re.search(r'example|usage', content, re.IGNORECASE)),
                "has_return_info": bool(re.search(r'return|output', content, re.IGNORECASE)),
                "has_error_handling": bool(re.search(r'error|exception', content, re.IGNORECASE))
            }
            
            completeness_score = sum(completeness_checks.values()) / len(completeness_checks) * 100
            validation_result["completeness_score"] = completeness_score
            
            # Add issues for missing sections
            for check, passed in completeness_checks.items():
                if not passed:
                    validation_result["issues"].append(f"Missing: {check.replace('has_', '').replace('_', ' ')}")
        
        # Check quality
        quality_checks = {
            "adequate_length": len(content) > 500,
            "has_code_examples": bool(re.search(r'```|<code>', content)),
            "has_structured_sections": content.count('#') > 3 or content.count('<h') > 3,
            "recent_generation": True  # Assume recent if we have it cached
        }
        
        quality_score = sum(quality_checks.values()) / len(quality_checks) * 100
        validation_result["quality_score"] = quality_score
        
        # Generate recommendations
        if validation_result["completeness_score"] < 80:
            validation_result["recommendations"].append("Add missing documentation sections")
        
        if validation_result["quality_score"] < 70:
            validation_result["recommendations"].append("Improve documentation quality with more examples and details")
        
        if not validation_result["issues"] and not validation_result["recommendations"]:
            validation_result["recommendations"].append("Documentation appears complete and well-structured")
        
        # Overall status
        overall_score = (validation_result["completeness_score"] + validation_result["quality_score"]) / 2
        
        if overall_score >= 80:
            status = "excellent"
        elif overall_score >= 60:
            status = "good"
        elif overall_score >= 40:
            status = "needs_improvement"
        else:
            status = "poor"
        
        return json.dumps({
            "status": "success",
            "validation_result": validation_result,
            "overall_status": status,
            "overall_score": overall_score,
            "message": f"Documentation validation completed - Status: {status}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Documentation validation failed: {str(e)}"
        })


@tool
def list_tool_documentation() -> str:
    """
    List all available tool documentation.
    
    Returns:
        JSON string with documentation list
    """
    try:
        doc_list = []
        
        for tool_name, doc_info in _docs_cache.items():
            doc_list.append({
                "tool_name": tool_name,
                "file_path": doc_info["file_path"],
                "format": doc_info["format"],
                "generated_at": doc_info["generated_at"],
                "last_updated": doc_info.get("last_updated"),
                "exists": os.path.exists(doc_info["file_path"])
            })
        
        return json.dumps({
            "status": "success",
            "documentation_list": doc_list,
            "total_count": len(doc_list),
            "message": f"Found documentation for {len(doc_list)} tools"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to list documentation: {str(e)}"
        })