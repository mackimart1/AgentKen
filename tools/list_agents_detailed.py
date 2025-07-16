from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import utils
import json


class ListAgentsDetailedInput(BaseModel):
    include_inactive: bool = Field(
        default=False, 
        description="Whether to include inactive agents in the listing"
    )
    filter_by_capability: Optional[str] = Field(
        default=None, 
        description="Filter agents by a specific capability (e.g., 'data_processing', 'web_research')"
    )
    format_output: str = Field(
        default="detailed", 
        description="Output format: 'detailed', 'summary', or 'json'"
    )


@tool(args_schema=ListAgentsDetailedInput)
def list_agents_detailed(
    include_inactive: bool = False, 
    filter_by_capability: Optional[str] = None,
    format_output: str = "detailed"
) -> str:
    """
    Lists all available agents with detailed information including capabilities, 
    descriptions, input/output schemas, and metadata.
    
    Args:
        include_inactive: Whether to include inactive agents
        filter_by_capability: Filter agents by specific capability
        format_output: Output format (detailed, summary, or json)
    
    Returns:
        Formatted string with agent information
    """
    try:
        # Get agents manifest
        agents_manifest = utils.get_agents_manifest()
        
        if not agents_manifest:
            return "No agents found in the manifest."
        
        # Filter agents
        filtered_agents = []
        for agent in agents_manifest:
            # Filter by status if not including inactive
            if not include_inactive and agent.get("status") == "inactive":
                continue
                
            # Filter by capability if specified
            if filter_by_capability:
                capabilities = agent.get("capabilities", [])
                if isinstance(capabilities, list) and filter_by_capability not in capabilities:
                    continue
                    
            filtered_agents.append(agent)
        
        if not filtered_agents:
            return f"No agents found matching the criteria."
        
        # Format output based on requested format
        if format_output == "json":
            return json.dumps(filtered_agents, indent=2)
        elif format_output == "summary":
            return _format_agents_summary(filtered_agents)
        else:  # detailed
            return _format_agents_detailed(filtered_agents)
            
    except Exception as e:
        return f"Error retrieving agent information: {str(e)}"


def _format_agents_summary(agents: List[Dict[str, Any]]) -> str:
    """Format agents in summary view."""
    output = f"=== AVAILABLE AGENTS ({len(agents)} total) ===\n\n"
    
    for agent in agents:
        name = agent.get("name", "Unknown")
        description = agent.get("description", "No description available")
        status = agent.get("status", "unknown")
        capabilities = agent.get("capabilities", [])
        
        output += f"• {name.upper()}\n"
        output += f"  Status: {status}\n"
        output += f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}\n"
        if capabilities:
            output += f"  Capabilities: {', '.join(capabilities[:3])}{'...' if len(capabilities) > 3 else ''}\n"
        output += "\n"
    
    return output


def _format_agents_detailed(agents: List[Dict[str, Any]]) -> str:
    """Format agents in detailed view."""
    output = f"=== DETAILED AGENT INFORMATION ({len(agents)} agents) ===\n\n"
    
    for i, agent in enumerate(agents, 1):
        name = agent.get("name", "Unknown")
        description = agent.get("description", "No description available")
        module_path = agent.get("module_path", "Unknown")
        function_name = agent.get("function_name", "Unknown")
        status = agent.get("status", "unknown")
        capabilities = agent.get("capabilities", [])
        author = agent.get("author", "Unknown")
        created_at = agent.get("created_at", "Unknown")
        version = agent.get("version", "Unknown")
        
        output += f"[{i}] {name.upper()}\n"
        output += f"    Description: {description}\n"
        output += f"    Module: {module_path}\n"
        output += f"    Function: {function_name}\n"
        output += f"    Status: {status}\n"
        
        if capabilities:
            output += f"    Capabilities: {', '.join(capabilities)}\n"
        
        if author != "Unknown":
            output += f"    Author: {author}\n"
        if created_at != "Unknown":
            output += f"    Created: {created_at}\n"
        if version != "Unknown":
            output += f"    Version: {version}\n"
        
        # Input schema
        input_schema = agent.get("input_schema", {})
        if input_schema:
            output += f"    Input Schema:\n"
            if "properties" in input_schema:
                for prop, details in input_schema["properties"].items():
                    prop_type = details.get("type", "unknown")
                    prop_desc = details.get("description", "No description")
                    required = prop in input_schema.get("required", [])
                    req_marker = " (required)" if required else " (optional)"
                    output += f"      - {prop}: {prop_type}{req_marker} - {prop_desc}\n"
        
        # Output schema
        output_schema = agent.get("output_schema", {})
        if output_schema:
            output += f"    Output Schema: {output_schema.get('type', 'object')}\n"
            if "properties" in output_schema:
                for prop, details in output_schema["properties"].items():
                    prop_type = details.get("type", "unknown")
                    output += f"      - {prop}: {prop_type}\n"
        
        output += "\n" + "="*60 + "\n\n"
    
    return output