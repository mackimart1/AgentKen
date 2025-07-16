from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import utils
import os
import json
from datetime import datetime


class SystemOverviewInput(BaseModel):
    include_stats: bool = Field(
        default=True, 
        description="Whether to include system statistics"
    )
    include_capabilities: bool = Field(
        default=True, 
        description="Whether to include capability analysis"
    )
    format_output: str = Field(
        default="detailed", 
        description="Output format: 'detailed', 'summary', or 'json'"
    )


@tool(args_schema=SystemOverviewInput)
def system_overview(
    include_stats: bool = True,
    include_capabilities: bool = True,
    format_output: str = "detailed"
) -> str:
    """
    Provides a comprehensive overview of the AgentK system including agents, 
    tools, capabilities, and system statistics.
    
    Args:
        include_stats: Whether to include system statistics
        include_capabilities: Whether to include capability analysis
        format_output: Output format (detailed, summary, or json)
    
    Returns:
        Formatted string with complete system overview
    """
    try:
        # Gather system data
        agents_manifest = utils.get_agents_manifest()
        tools_manifest = utils.get_tools_manifest()
        
        system_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": agents_manifest,
            "tools": tools_manifest,
            "stats": {},
            "capabilities": {}
        }
        
        if include_stats:
            system_data["stats"] = _gather_system_stats(agents_manifest, tools_manifest)
        
        if include_capabilities:
            system_data["capabilities"] = _analyze_capabilities(agents_manifest, tools_manifest)
        
        # Format output
        if format_output == "json":
            return json.dumps(system_data, indent=2)
        elif format_output == "summary":
            return _format_system_summary(system_data)
        else:  # detailed
            return _format_system_detailed(system_data)
            
    except Exception as e:
        return f"Error generating system overview: {str(e)}"


def _gather_system_stats(agents_manifest: List[Dict], tools_manifest: List[Dict]) -> Dict[str, Any]:
    """Gather system statistics."""
    stats = {
        "total_agents": len(agents_manifest),
        "total_tools": len(tools_manifest),
        "active_agents": 0,
        "inactive_agents": 0,
        "active_tools": 0,
        "inactive_tools": 0,
        "agents_by_author": {},
        "tools_by_author": {},
        "recent_additions": {"agents": [], "tools": []}
    }
    
    # Analyze agents
    for agent in agents_manifest:
        status = agent.get("status", "active")
        if status == "active":
            stats["active_agents"] += 1
        else:
            stats["inactive_agents"] += 1
        
        author = agent.get("author", "Unknown")
        stats["agents_by_author"][author] = stats["agents_by_author"].get(author, 0) + 1
        
        # Check if recent (within last 30 days - simplified check)
        created_at = agent.get("created_at", "")
        if created_at and "2025" in created_at:  # Simple recent check
            stats["recent_additions"]["agents"].append(agent.get("name", "Unknown"))
    
    # Analyze tools
    for tool in tools_manifest:
        status = tool.get("status", "active")
        if status == "active":
            stats["active_tools"] += 1
        else:
            stats["inactive_tools"] += 1
        
        author = tool.get("author", "Unknown")
        stats["tools_by_author"][author] = stats["tools_by_author"].get(author, 0) + 1
        
        # Check if recent
        created_at = tool.get("created_at", "")
        if created_at and "2025" in created_at:  # Simple recent check
            stats["recent_additions"]["tools"].append(tool.get("name", "Unknown"))
    
    return stats


def _analyze_capabilities(agents_manifest: List[Dict], tools_manifest: List[Dict]) -> Dict[str, Any]:
    """Analyze system capabilities."""
    capabilities = {
        "agent_capabilities": {},
        "tool_categories": {},
        "coverage_analysis": {},
        "capability_gaps": []
    }
    
    # Analyze agent capabilities
    all_agent_capabilities = set()
    for agent in agents_manifest:
        agent_caps = agent.get("capabilities", [])
        if isinstance(agent_caps, list):
            for cap in agent_caps:
                all_agent_capabilities.add(cap)
                if cap not in capabilities["agent_capabilities"]:
                    capabilities["agent_capabilities"][cap] = []
                capabilities["agent_capabilities"][cap].append(agent.get("name", "Unknown"))
    
    # Categorize tools (basic categorization based on name patterns)
    tool_categories = {
        "file_management": [],
        "code_execution": [],
        "web_research": [],
        "data_processing": [],
        "system_management": [],
        "communication": [],
        "other": []
    }
    
    for tool in tools_manifest:
        tool_name = tool.get("name", "").lower()
        categorized = False
        
        if any(keyword in tool_name for keyword in ["file", "read", "write", "delete"]):
            tool_categories["file_management"].append(tool.get("name"))
            categorized = True
        elif any(keyword in tool_name for keyword in ["code", "execute", "run", "shell"]):
            tool_categories["code_execution"].append(tool.get("name"))
            categorized = True
        elif any(keyword in tool_name for keyword in ["web", "search", "fetch", "duck"]):
            tool_categories["web_research"].append(tool.get("name"))
            categorized = True
        elif any(keyword in tool_name for keyword in ["data", "process", "validate"]):
            tool_categories["data_processing"].append(tool.get("name"))
            categorized = True
        elif any(keyword in tool_name for keyword in ["list", "assign", "agent", "system"]):
            tool_categories["system_management"].append(tool.get("name"))
            categorized = True
        elif any(keyword in tool_name for keyword in ["terminal", "session"]):
            tool_categories["communication"].append(tool.get("name"))
            categorized = True
        
        if not categorized:
            tool_categories["other"].append(tool.get("name"))
    
    capabilities["tool_categories"] = {k: v for k, v in tool_categories.items() if v}
    
    # Basic coverage analysis
    core_capabilities = [
        "file_management", "code_execution", "web_research", 
        "data_processing", "system_monitoring", "debugging"
    ]
    
    coverage = {}
    for cap in core_capabilities:
        agent_coverage = cap in all_agent_capabilities
        tool_coverage = cap in capabilities["tool_categories"]
        coverage[cap] = {
            "agent_support": agent_coverage,
            "tool_support": tool_coverage,
            "full_coverage": agent_coverage and tool_coverage
        }
    
    capabilities["coverage_analysis"] = coverage
    
    # Identify potential gaps
    gaps = []
    for cap, cov in coverage.items():
        if not cov["full_coverage"]:
            if not cov["agent_support"] and not cov["tool_support"]:
                gaps.append(f"No {cap} capability (missing both agents and tools)")
            elif not cov["agent_support"]:
                gaps.append(f"No {cap} agents (tools available)")
            elif not cov["tool_support"]:
                gaps.append(f"No {cap} tools (agents available)")
    
    capabilities["capability_gaps"] = gaps
    
    return capabilities


def _format_system_summary(system_data: Dict[str, Any]) -> str:
    """Format system data in summary view."""
    stats = system_data.get("stats", {})
    capabilities = system_data.get("capabilities", {})
    
    output = "=== AGENTK SYSTEM OVERVIEW (SUMMARY) ===\n\n"
    
    # Basic stats
    output += f"📊 SYSTEM STATISTICS:\n"
    output += f"   • Total Agents: {stats.get('total_agents', 0)} ({stats.get('active_agents', 0)} active)\n"
    output += f"   • Total Tools: {stats.get('total_tools', 0)} ({stats.get('active_tools', 0)} active)\n"
    
    # Recent additions
    recent_agents = stats.get("recent_additions", {}).get("agents", [])
    recent_tools = stats.get("recent_additions", {}).get("tools", [])
    if recent_agents or recent_tools:
        output += f"\n🆕 RECENT ADDITIONS:\n"
        if recent_agents:
            output += f"   • New Agents: {', '.join(recent_agents[:3])}{'...' if len(recent_agents) > 3 else ''}\n"
        if recent_tools:
            output += f"   • New Tools: {', '.join(recent_tools[:3])}{'...' if len(recent_tools) > 3 else ''}\n"
    
    # Capabilities overview
    agent_caps = capabilities.get("agent_capabilities", {})
    tool_cats = capabilities.get("tool_categories", {})
    
    if agent_caps:
        output += f"\n🎯 TOP AGENT CAPABILITIES:\n"
        sorted_caps = sorted(agent_caps.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for cap, agents in sorted_caps:
            output += f"   • {cap}: {len(agents)} agent(s)\n"
    
    if tool_cats:
        output += f"\n🔧 TOOL CATEGORIES:\n"
        for category, tools in tool_cats.items():
            output += f"   • {category.replace('_', ' ').title()}: {len(tools)} tool(s)\n"
    
    # Capability gaps
    gaps = capabilities.get("capability_gaps", [])
    if gaps:
        output += f"\n⚠️  CAPABILITY GAPS:\n"
        for gap in gaps[:3]:
            output += f"   • {gap}\n"
        if len(gaps) > 3:
            output += f"   • ... and {len(gaps) - 3} more\n"
    
    return output


def _format_system_detailed(system_data: Dict[str, Any]) -> str:
    """Format system data in detailed view."""
    stats = system_data.get("stats", {})
    capabilities = system_data.get("capabilities", {})
    timestamp = system_data.get("timestamp", "Unknown")
    
    output = "=== AGENTK SYSTEM OVERVIEW (DETAILED) ===\n"
    output += f"Generated: {timestamp}\n\n"
    
    # Detailed statistics
    output += "📊 DETAILED SYSTEM STATISTICS:\n"
    output += f"   Total Agents: {stats.get('total_agents', 0)}\n"
    output += f"   ├─ Active: {stats.get('active_agents', 0)}\n"
    output += f"   └─ Inactive: {stats.get('inactive_agents', 0)}\n\n"
    
    output += f"   Total Tools: {stats.get('total_tools', 0)}\n"
    output += f"   ├─ Active: {stats.get('active_tools', 0)}\n"
    output += f"   └─ Inactive: {stats.get('inactive_tools', 0)}\n\n"
    
    # Authors breakdown
    agents_by_author = stats.get("agents_by_author", {})
    tools_by_author = stats.get("tools_by_author", {})
    
    if agents_by_author:
        output += "👥 AGENTS BY AUTHOR:\n"
        for author, count in sorted(agents_by_author.items(), key=lambda x: x[1], reverse=True):
            output += f"   • {author}: {count} agent(s)\n"
        output += "\n"
    
    if tools_by_author:
        output += "🔨 TOOLS BY AUTHOR:\n"
        for author, count in sorted(tools_by_author.items(), key=lambda x: x[1], reverse=True):
            output += f"   • {author}: {count} tool(s)\n"
        output += "\n"
    
    # Detailed capabilities
    agent_caps = capabilities.get("agent_capabilities", {})
    if agent_caps:
        output += "🎯 AGENT CAPABILITIES BREAKDOWN:\n"
        for cap, agents in sorted(agent_caps.items()):
            output += f"   • {cap}:\n"
            for agent in agents:
                output += f"     - {agent}\n"
        output += "\n"
    
    # Tool categories
    tool_cats = capabilities.get("tool_categories", {})
    if tool_cats:
        output += "🔧 TOOL CATEGORIES BREAKDOWN:\n"
        for category, tools in tool_cats.items():
            output += f"   • {category.replace('_', ' ').title()}:\n"
            for tool in tools:
                output += f"     - {tool}\n"
        output += "\n"
    
    # Coverage analysis
    coverage = capabilities.get("coverage_analysis", {})
    if coverage:
        output += "📈 CAPABILITY COVERAGE ANALYSIS:\n"
        for cap, cov in coverage.items():
            status = "✅ Full" if cov["full_coverage"] else "⚠️  Partial" if (cov["agent_support"] or cov["tool_support"]) else "❌ None"
            output += f"   • {cap.replace('_', ' ').title()}: {status}\n"
            output += f"     - Agent Support: {'Yes' if cov['agent_support'] else 'No'}\n"
            output += f"     - Tool Support: {'Yes' if cov['tool_support'] else 'No'}\n"
        output += "\n"
    
    # Capability gaps
    gaps = capabilities.get("capability_gaps", [])
    if gaps:
        output += "⚠️  IDENTIFIED CAPABILITY GAPS:\n"
        for i, gap in enumerate(gaps, 1):
            output += f"   {i}. {gap}\n"
        output += "\n"
    
    return output