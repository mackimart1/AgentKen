"""
ML Engineer Agent: Designs machine learning models, assesses data requirements,
requests necessary tools, and reports on feasibility and performance.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import logging

logger = logging.getLogger(__name__)

# Import necessary tools for this agent
# Note: This agent's role includes identifying *new* tools needed for ML tasks
# and requesting them via assign_agent_to_task targeting 'tool_maker'.
# Assuming these tools exist in the 'tools' directory as per AgentK structure
try:
    from tools.assign_agent_to_task import assign_agent_to_task
    from tools.read_file import read_file
    from tools.write_to_file import write_to_file

    tools = [assign_agent_to_task, read_file, write_to_file]
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all tools for ml_engineer: {e}")
    # Define tools as an empty list or handle appropriately if imports fail
    tools = []  # Agent might be non-functional if tools are missing
    TOOLS_AVAILABLE = False

system_prompt = """You are ml_engineer, a ReAct agent responsible for designing machine learning models,
especially neural networks. Your tasks include:

1.  **Analyze Requirements:** Understand the goal, data availability (or requirements), and constraints.
2.  **Design Model:** Propose a suitable model architecture, data processing steps, and evaluation strategy.
3.  **Identify Tool Gaps:** Determine if specialized tools (e.g., for data preprocessing, model training, evaluation) are needed. If so, use the 'assign_agent_to_task' tool to request the 'tool_maker' agent to build them. Clearly specify the tool's function, inputs, and outputs.
4.  **Assess Feasibility:** Report on the likelihood of success, potential challenges, and resource estimates.
5.  **Document:** Use 'write_to_file' to save design documents or reports.

You do NOT train or evaluate models directly unless specific tools for those actions are available and requested.
Your primary role is planning, design, and coordination with tool_maker.
"""


def reasoning(state: MessagesState) -> dict:
    """
    ML Engineer's reasoning step.
    """
    print("ml_engineer is thinking...")
    messages = state["messages"]
    # Ensure tools were loaded before binding
    if not TOOLS_AVAILABLE:
        # Handle case where tools failed to load - maybe return an error message
        # For now, proceeding without tools bound, which might limit functionality
        print("Warning: ml_engineer proceeding without tools bound.")
        tooled_up_model = config.default_langchain_model
    else:
        # Use Google Gemini for tool calling from hybrid configuration
        tool_model = config.get_model_for_tools()
        if tool_model is None:
            # Fallback to default model if hybrid setup fails
            tool_model = config.default_langchain_model
            logger.warning("Using fallback model for tools - may not support function calling")
        
        tooled_up_model = tool_model.bind_tools(tools)

    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}


def check_for_tool_calls(state: MessagesState) -> str:
    """
    Checks the last message for tool calls.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Only proceed to tools node if tools were successfully loaded and called
    if (
        TOOLS_AVAILABLE
        and isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        # Safely handle content which might be a string or list
        content = last_message.content
        if content:
            if isinstance(content, str) and content.strip():
                print("ml_engineer thought this:")
                print(content)
            elif isinstance(content, list) and content:
                print("ml_engineer thought this:")
                print(content)
        print()
        print("ml_engineer is acting by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"

    return END


# Ensure acting node is only added if tools are available
if TOOLS_AVAILABLE:
    acting = ToolNode(tools)
else:
    # Define a dummy acting node or handle the graph structure differently
    # This prevents errors if 'tools' is empty but 'acting' is referenced.
    # A simple approach is to make 'acting' do nothing or raise an error.
    def dummy_acting(state: MessagesState) -> dict:
        print("Error: ml_engineer cannot act, tools failed to load.")
        # Return state unchanged or add an error message
        return {}  # Or {"messages": [AIMessage(content="Tool loading failed.")]}

    acting = dummy_acting


workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
# Only add the acting node and edges if tools were loaded
if TOOLS_AVAILABLE:
    workflow.add_node("tools", acting)
    workflow.set_entry_point("reasoning")
    workflow.add_conditional_edges(
        "reasoning",
        check_for_tool_calls,
    )
    workflow.add_edge("tools", "reasoning")
else:
    # If no tools, reasoning directly leads to END
    workflow.set_entry_point("reasoning")
    workflow.add_edge("reasoning", END)


graph = workflow.compile()


def ml_engineer(task: str) -> dict:
    """
    Executes the ML Engineer workflow.

    Args:
        task (str): The ML design/feasibility task.

    Returns:
        dict: A dictionary containing status, result, and message.
    """
    # Check if graph compilation might have failed due to missing tools
    if not graph:
        return {
            "status": "failure",
            "result": None,
            "message": "ML Engineer agent could not be initialized properly (graph compilation failed, likely due to missing tools).",
        }

    try:
        final_state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task),
                ]
            }
        )
    except Exception as e:
        print(f"Error invoking ml_engineer graph: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"ML Engineer agent failed during execution: {e}",
        }

    last_message_content = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_message = final_state["messages"][-1]
        if isinstance(last_message, (AIMessage, HumanMessage)):
            last_message_content = last_message.content
        elif isinstance(last_message, ToolMessage):
            # Ensure content is stringified, as it might be complex data
            last_message_content = str(last_message.content)
        elif isinstance(last_message, list):  # Handle potential list wrapping
            try:
                # Access content of the last message if it's a list of messages
                last_msg_in_list = last_message[-1]
                if hasattr(last_msg_in_list, "content"):
                    last_message_content = last_msg_in_list.content
                else:
                    last_message_content = str(last_msg_in_list)  # Fallback
            except (IndexError, AttributeError, TypeError):
                last_message_content = str(
                    last_message
                )  # Fallback string representation
        else:
            # Fallback for any other unexpected type
            last_message_content = str(last_message)

    # Basic status determination - refine as needed
    status = "success" if last_message_content else "failure"
    # More robust failure check
    # Ensure last_message_content is a string for string operations
    message_str = str(last_message_content) if last_message_content else ""
    if (
        "failed" in message_str.lower()
        or "unable to" in message_str.lower()
        or status == "failure"
    ):
        status = "failure"
        result_data = None
    else:
        # Result could be a path to a design doc, a feasibility summary, etc.
        result_data = last_message_content

    print(
        f"ML Engineer finished task. Status: {status}. Final Message: {last_message_content}"
    )

    return {"status": status, "result": result_data, "message": last_message_content}


# Example of how to potentially use the agent (for testing/dev)
# if __name__ == '__main__':
#     # This part would only run if the script is executed directly
#     # Requires config.py setup and necessary libraries installed
#     print("Testing ml_engineer agent...")
#     test_task = "Assess the feasibility of using a transformer model for time series forecasting with potentially missing data. What tools would be needed?"
#     output = ml_engineer(test_task)
#     print("--- Output ---")
#     import json
#     print(json.dumps(output, indent=2))
#     print("--------------")
