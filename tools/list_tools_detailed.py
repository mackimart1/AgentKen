from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import utils
import json


class ListToolsDetailedInput(BaseModel):
    include_inactive: bool = Field(
        default=False, 
        description="Whether to include inactive tools in the listing"
    )
    filter_by_category: Optional[str] = Field(
        default=None, 
        description="Filter tools by category (e.g., 'file_management', 'web_research', 'code_execution')"
    )
    format_output: str = Field(
        default="detailed", 
        description="Output format: 'detailed', 'summary', or 'json'"
    )


@tool(args_schema=ListToolsDetailedInput)
def list_tools_detailed(
    include_inactive: bool = False, 
    filter_by_category: Optional[str] = None,
    format_output: str = "detailed"
) -> str:
    """
    Lists all available tools with detailed information including descriptions, 
    input/output schemas, parameters, and metadata.
    
    Args:
        include_inactive: Whether to include inactive tools
        filter_by_category: Filter tools by specific category
        format_output: Output format (detailed, summary, or json)
    
    Returns:
        Formatted string with tool information
    """
    try:
        # Get tools manifest
        tools_manifest = utils.get_tools_manifest()
        
        if not tools_manifest:
            return "No tools found in the manifest."
        
        # Filter tools
        filtered_tools = []
        for tool in tools_manifest:
            # Filter by status if not including inactive
            if not include_inactive and tool.get("status") == "inactive":
                continue
                
            # Filter by category if specified (this would need to be added to manifest)
            if filter_by_category:
                category = tool.get("category", "")
                if category != filter_by_category:
                    continue
                    
            filtered_tools.append(tool)
        
        if not filtered_tools:
            return f"No tools found matching the criteria."
        
        # Format output based on requested format
        if format_output == "json":
            return json.dumps(filtered_tools, indent=2)
        elif format_output == "summary":
            return _format_tools_summary(filtered_tools)
        else:  # detailed
            return _format_tools_detailed(filtered_tools)
            
    except Exception as e:
        return f"Error retrieving tool information: {str(e)}"


def _format_tools_summary(tools: List[Dict[str, Any]]) -> str:
    """Format tools in summary view."""
    output = f"=== AVAILABLE TOOLS ({len(tools)} total) ===\n\n"
    
    for tool in tools:
        name = tool.get("name", "Unknown")
        description = tool.get("description", "No description available")
        status = tool.get("status", "active")
        
        output += f"• {name}\n"
        output += f"  Status: {status}\n"
        output += f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}\n"
        output += "\n"
    
    return output


def _format_tools_detailed(tools: List[Dict[str, Any]]) -> str:
    """Format tools in detailed view."""
    output = f"=== DETAILED TOOL INFORMATION ({len(tools)} tools) ===\n\n"
    
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "Unknown")
        description = tool.get("description", "No description available")
        module_path = tool.get("module_path", "Unknown")
        function_name = tool.get("function_name", "Unknown")
        status = tool.get("status", "active")
        author = tool.get("author", "Unknown")
        created_at = tool.get("created_at", "Unknown")
        version = tool.get("version", "Unknown")
        
        output += f"[{i}] {name.upper()}\n"
        output += f"    Description: {description}\n"
        output += f"    Module: {module_path}\n"
        output += f"    Function: {function_name}\n"
        output += f"    Status: {status}\n"
        
        if author != "Unknown":
            output += f"    Author: {author}\n"
        if created_at != "Unknown":
            output += f"    Created: {created_at}\n"
        if version != "Unknown":
            output += f"    Version: {version}\n"
        
        # Input schema
        input_schema = tool.get("input_schema", {})
        if input_schema:
            output += f"    Input Parameters:\n"
            if "properties" in input_schema:
                for prop, details in input_schema["properties"].items():
                    prop_type = details.get("type", "unknown")
                    prop_desc = details.get("description", "No description")
                    default_val = details.get("default", None)
                    required = prop in input_schema.get("required", [])
                    
                    req_marker = " (required)" if required else " (optional)"
                    default_marker = f" [default: {default_val}]" if default_val is not None else ""
                    
                    output += f"      - {prop}: {prop_type}{req_marker}{default_marker}\n"
                    output += f"        {prop_desc}\n"
            elif input_schema == {}:
                output += f"      No parameters required\n"
        
        # Output schema
        output_schema = tool.get("output_schema", {})
        if output_schema:
            output += f"    Output Schema:\n"
            if "properties" in output_schema:
                for prop, details in output_schema["properties"].items():
                    prop_type = details.get("type", "unknown")
                    prop_desc = details.get("description", "")
                    output += f"      - {prop}: {prop_type}"
                    if prop_desc:
                        output += f" - {prop_desc}"
                    output += "\n"
            else:
                output_type = output_schema.get("type", "unknown")
                output += f"      Type: {output_type}\n"
        
        # Legacy parameters format (for older tools)
        parameters = tool.get("parameters", {})
        if parameters and not input_schema:
            output += f"    Parameters (legacy format):\n"
            for param, param_type in parameters.items():
                output += f"      - {param}: {param_type}\n"
        
        # Return type (legacy format)
        return_type = tool.get("return_type")
        if return_type and not output_schema:
            output += f"    Return Type: {return_type}\n"
        
        output += "\n" + "="*60 + "\n\n"
    
    return output