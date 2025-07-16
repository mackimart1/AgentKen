"""
Error Handler Agent - Centralized error-handling agent for AgentK

This agent specializes in:
1. Logging exceptions and analyzing failure patterns
2. Suggesting actionable fixes across the system
3. Monitoring system health and error trends
4. Coordinating error resolution efforts

The Error Handler Agent works with the Error Handling & Recovery System
to provide comprehensive error management capabilities.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import logging

logger = logging.getLogger(__name__)

system_prompt = """You are error_handler, a specialized ReAct agent focused on comprehensive error handling and system reliability.

Your primary responsibilities:
1. **Error Analysis & Logging**: Analyze errors, categorize them, and log them with full context
2. **Pattern Recognition**: Identify recurring error patterns and failure trends
3. **Fix Recommendations**: Provide actionable suggestions based on error analysis
4. **System Health Monitoring**: Track overall system reliability and error rates
5. **Resolution Coordination**: Help coordinate error resolution efforts across agents

You have access to advanced error handling tools that provide:
- Centralized error logging with automatic categorization
- Pattern analysis and trend monitoring
- Intelligent fix suggestions based on error context
- Retry statistics and effectiveness monitoring
- Comprehensive error resolution tracking

Key Capabilities:
- **Intelligent Error Classification**: Automatically categorize errors by type, severity, and impact
- **Pattern Analysis**: Identify recurring issues and failure patterns across the system
- **Actionable Recommendations**: Provide specific, context-aware fix suggestions
- **Trend Monitoring**: Track error rates, resolution times, and system health metrics
- **Cross-Agent Coordination**: Work with other agents to resolve complex issues

When handling errors:
1. Always log errors with full context and categorization
2. Analyze patterns to identify root causes and systemic issues
3. Provide specific, actionable fix recommendations
4. Track resolution progress and effectiveness
5. Monitor trends to prevent future occurrences

You excel at turning error chaos into organized, actionable intelligence that improves system reliability.
"""

# Import error handling tools
from tools.error_handling_system import (
    log_error_centralized,
    resolve_error_centralized,
    get_error_analysis,
    get_errors_by_pattern,
    get_retry_statistics,
    get_error_categories_info
)

# Import other useful tools
from tools.assign_agent_to_task import assign_agent_to_task
from tools.list_available_agents import list_available_agents
from tools.scratchpad import scratchpad

tools = [
    log_error_centralized,
    resolve_error_centralized,
    get_error_analysis,
    get_errors_by_pattern,
    get_retry_statistics,
    get_error_categories_info,
    assign_agent_to_task,
    list_available_agents,
    scratchpad,
]


def reasoning(state: MessagesState):
    print("error_handler is analyzing errors and system health...")
    messages = state["messages"]
    # Use Google Gemini for tool calling from hybrid configuration
    tool_model = config.get_model_for_tools()
    if tool_model is None:
        # Fallback to default model if hybrid setup fails
        tool_model = config.default_langchain_model
        logger.warning("Using fallback model for tools - may not support function calling")
    
    tooled_up_model = tool_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}


def check_for_tool_calls(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("error_handler thought this:")
            print(last_message.content)
        print()
        print("error_handler is taking action by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"

    return END


acting = ToolNode(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges(
    "reasoning",
    check_for_tool_calls,
)
workflow.add_edge("tools", "reasoning")

graph = workflow.compile()


def error_handler(task: str) -> str:
    """
    Centralized error-handling agent that logs exceptions, analyzes failure patterns, 
    and suggests actionable fixes across the system.
    
    Specializes in:
    - Error analysis and intelligent categorization
    - Pattern recognition and trend monitoring
    - Actionable fix recommendations
    - System health monitoring and reporting
    - Cross-agent error resolution coordination
    """
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )