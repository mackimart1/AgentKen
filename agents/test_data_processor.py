"""
Test Data Processor: Processes and analyzes test data

Created by: Test Agent Smith
User ID: agent_smith_user
Created: 2025-07-03 21:09:19

Capabilities:
    - data_processing
    - analysis
    - reporting

This agent follows the standard AgentK pattern and integrates with the
permissioned creation system.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import utils
import memory_manager

# Setup logger for test_data_processor
logger = logging.getLogger(__name__)

# Initialize memory manager
try:
    memory_manager_instance = memory_manager.MemoryManager()
    MEMORY_AVAILABLE = True
except Exception as e:
    logger.warning(f"Memory manager not available: {e}")
    memory_manager_instance = None
    MEMORY_AVAILABLE = False


system_prompt = """You are test_data_processor, a ReAct agent in the AgentK system.

DESCRIPTION:
Processes and analyzes test data

CAPABILITIES:
    - data_processing
    - analysis
    - reporting

WORKFLOW:
1. Analyze the task requirements thoroughly
2. Plan the approach and identify required tools
3. Execute the task using available tools
4. Validate results and provide comprehensive output
5. Handle errors gracefully with detailed error messages

RESPONSE FORMAT:
- Use inner monologue for planning and debugging thoughts
- Provide clear success/failure status
- Include detailed error messages when failures occur
- Follow the standard AgentK response format

ERROR HANDLING:
- Always catch and log exceptions
- Provide meaningful error messages
- Return structured error responses
- Include retry logic where appropriate
"""

# Get available tools
tools = utils.all_tool_functions()


def get_memory_context(task_description: str) -> str:
    """Retrieve relevant memories for the task"""
    if not MEMORY_AVAILABLE or not memory_manager_instance:
        return ""

    try:
        relevant_memories = memory_manager_instance.retrieve_memories(
            query_text=task_description, k=5
        )
        if relevant_memories:
            return (
                "Relevant context from past interactions:\n"
                + "\n".join([f"- {mem}" for mem in relevant_memories])
                + "\n\n---\n\n"
            )
    except Exception as e:
        logger.warning(f"Failed to retrieve memories: {e}")

    return ""


def reasoning(state: MessagesState) -> Dict[str, Any]:
    """Reasoning node for the agent workflow"""
    messages = state["messages"]
    last_message = messages[-1]

    # Get task description
    if isinstance(last_message, HumanMessage):
        task_description = last_message.content
    else:
        task_description = "Unknown task"

    # Get memory context
    memory_context = get_memory_context(task_description)

    # Build context for the agent
    context = f"""{memory_context}
TASK: {task_description}

AVAILABLE TOOLS:
{', '.join([tool.name for tool in tools])}

Please analyze this task and determine the best approach to complete it.
Use the available tools as needed and provide a comprehensive response.
"""

    # Create system message with context
    system_msg = SystemMessage(content=system_prompt + "\n\n" + context)

    return {"messages": [system_msg, HumanMessage(content=task_description)]}


def acting(state: MessagesState) -> Dict[str, Any]:
    """Acting node for tool execution"""
    return {"messages": state["messages"]}


def check_for_tool_calls(state: MessagesState) -> str:
    """Check if the last message contains tool calls"""
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check if the message content indicates completion
    content = last_message.content.lower() if hasattr(last_message, "content") else ""
    if any(
        keyword in content for keyword in ["success", "completed", "finished", "done"]
    ):
        return END

    return "reasoning"


def analyze_result(messages_history: list) -> tuple[str, Optional[str], str]:
    """Analyze the agent execution result"""
    if not messages_history:
        return "failure", None, "No messages generated"

    last_message = messages_history[-1]
    content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    # Check for success indicators
    success_indicators = ["success", "completed", "finished", "done"]
    failure_indicators = ["error", "failed", "failure", "exception"]

    content_lower = content.lower()

    if any(indicator in content_lower for indicator in success_indicators):
        # Extract result name if possible
        result_match = re.search(r"['\"]([^'\"]+)['\"]", content)
        result_name = result_match.group(1) if result_match else None

        return "success", result_name, content

    elif any(indicator in content_lower for indicator in failure_indicators):
        return "failure", None, content

    else:
        # Default to success if no clear indicators
        return "success", None, content


def test_data_processor(task: str) -> Dict[str, Any]:
    """
    Test Data Processor agent implementation.

    Args:
        task (str): The task to perform

    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
    """
    try:
        logger.info(f"Starting test_data_processor task: {task}")

        # Create workflow
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("reasoning", reasoning)
        workflow.add_node("tools", ToolNode(tools))

        # Add edges
        workflow.add_conditional_edges("reasoning", check_for_tool_calls)
        workflow.add_edge("tools", "reasoning")

        # Set entry point
        workflow.set_entry_point("reasoning")

        # Compile workflow
        app = workflow.compile()

        # Execute workflow
        result = app.invoke({"messages": [HumanMessage(content=task)]})

        # Analyze result
        messages_history = result.get("messages", [])
        status, result_name, message = analyze_result(messages_history)

        logger.info(f"test_data_processor completed with status: {status}")

        return {"status": status, "result": result_name, "message": message}

    except Exception as e:
        logger.error(f"Error in test_data_processor: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to complete task: {str(e)}",
        }


if __name__ == "__main__":
    # Test the agent
    test_task = "Test task for Test Data Processor"
    result = test_data_processor(test_task)
    print(f"Test result: {result}")
