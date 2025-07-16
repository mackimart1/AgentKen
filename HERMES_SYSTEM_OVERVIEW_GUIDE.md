# Hermes System Overview Tools Guide

This guide explains the new tools available to Hermes for viewing and understanding the AgentK system.

## New Tools Available

### 1. `system_overview`
**Purpose**: Provides a comprehensive overview of the entire AgentK system

**Parameters**:
- `include_stats` (boolean, default: true) - Include system statistics
- `include_capabilities` (boolean, default: true) - Include capability analysis  
- `format_output` (string, default: "detailed") - Output format: "detailed", "summary", or "json"

**Use Cases**:
- Get a high-level view of the entire system
- Understand system capabilities and gaps
- Check system health and statistics
- Identify recent additions to the system

**Example Usage**:
```
system_overview(format_output="summary")  # Quick overview
system_overview(format_output="detailed") # Full detailed analysis
```

### 2. `list_agents_detailed`
**Purpose**: Lists all available agents with detailed information

**Parameters**:
- `include_inactive` (boolean, default: false) - Include inactive agents
- `filter_by_capability` (string, optional) - Filter by specific capability
- `format_output` (string, default: "detailed") - Output format: "detailed", "summary", or "json"

**Use Cases**:
- Find agents with specific capabilities
- Get detailed information about agent schemas
- Understand what agents are available for task assignment
- Check agent metadata (author, version, creation date)

**Example Usage**:
```
list_agents_detailed(format_output="summary")  # Quick agent list
list_agents_detailed(filter_by_capability="code_development")  # Find coding agents
list_agents_detailed(include_inactive=true)  # Include all agents
```

### 3. `list_tools_detailed`
**Purpose**: Lists all available tools with detailed information

**Parameters**:
- `include_inactive` (boolean, default: false) - Include inactive tools
- `filter_by_category` (string, optional) - Filter by tool category
- `format_output` (string, default: "detailed") - Output format: "detailed", "summary", or "json"

**Use Cases**:
- Understand what tools are available
- Check tool parameters and schemas
- Find tools for specific categories
- Verify tool availability before planning

**Example Usage**:
```
list_tools_detailed(format_output="summary")  # Quick tool overview
list_tools_detailed(filter_by_category="file_management")  # Find file tools
```

## When to Use These Tools

### During Planning Phase
- Use `system_overview` to understand current system capabilities
- Use `list_agents_detailed` to find appropriate agents for tasks
- Check capability gaps to identify if new agents/tools are needed

### Before Agent Assignment
- Use `list_agents_detailed` with capability filters to find the best agent
- Verify agent input/output schemas match your task requirements
- Check agent status to ensure it's active

### For System Analysis
- Use `system_overview` with detailed format for comprehensive analysis
- Identify capability gaps that might need addressing
- Check system statistics and recent additions

### For Troubleshooting
- Use detailed views to understand why certain capabilities might be missing
- Check agent and tool statuses
- Verify manifest information is correct

## Integration with Existing Workflow

These tools complement your existing tools:
- Use alongside `list_available_agents` for basic agent listing
- Use before `assign_agent_to_task` to make informed decisions
- Use with `predict_agent` for data-driven agent selection

## Output Formats

### Summary Format
- Concise, easy-to-read overview
- Key statistics and highlights
- Good for quick decision making

### Detailed Format  
- Complete information including schemas
- Full capability listings
- Metadata and technical details
- Best for thorough analysis

### JSON Format
- Machine-readable structured data
- Useful for programmatic processing
- Can be stored or passed to other tools

## Best Practices

1. **Start with Summary**: Use summary format first to get an overview
2. **Filter When Needed**: Use capability/category filters to narrow results
3. **Check Before Assigning**: Always verify agent capabilities before task assignment
4. **Monitor System Health**: Regularly use system_overview to check for gaps
5. **Stay Updated**: Check for new agents and tools periodically

## Example Workflow

```
1. system_overview(format_output="summary")
   → Get overall system status

2. list_agents_detailed(filter_by_capability="relevant_capability")
   → Find suitable agents

3. assign_agent_to_task(agent_name="chosen_agent", task="specific_task")
   → Execute the task

4. system_overview(format_output="summary") 
   → Verify system state after task completion
```

These tools provide Hermes with comprehensive visibility into the AgentK system, enabling better decision-making and more effective task orchestration.