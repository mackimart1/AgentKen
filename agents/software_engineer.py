from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import logging

logger = logging.getLogger(__name__)

system_prompt = """You are software_engineer, a ReAct agent specialized in software development tasks.

Your capabilities include:
- Creating, reading, modifying, and deleting files
- Writing and executing code in various programming languages
- Running shell commands for development tasks (compilation, testing, package management, etc.)
- Listing and navigating directory structures
- Collaborating with other agents by assigning them specialized tasks
- Executing code securely to test functionality

You should approach tasks systematically:
1. Understand the requirements thoroughly
2. Plan your approach and identify necessary files/tools
3. Implement solutions step by step
4. Test your code when appropriate
5. Provide clear explanations of what you've accomplished

Always write clean, well-documented code following best practices for the target language.
When working with existing codebases, maintain consistency with existing patterns and styles.
"""

from tools.write_to_file import write_to_file
from tools.delete_file import delete_file
from tools.read_file import read_file
from tools.run_shell_command import run_shell_command
from tools.assign_agent_to_task import assign_agent_to_task
from tools.list_available_agents import list_available_agents
from tools.list_directory import list_directory
from tools.secure_code_executor import secure_code_executor

tools = [
    write_to_file,
    delete_file,
    read_file,
    run_shell_command,
    assign_agent_to_task,
    list_available_agents,
    list_directory,
    secure_code_executor,
]


def reasoning(state: MessagesState):
    print("software_engineer is thinking...")
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


def check_for_tool_calls(state: MessagesState) -> Literal["tools", "END"]:
    messages = state["messages"]
    last_message = messages[-1]

    # Only AIMessage has tool_calls attribute
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Handle content that could be string or list
        content = last_message.content
        if isinstance(content, str) and content.strip() != "":
            print("software_engineer thought this:")
            print(content)
        print()
        print("software_engineer is acting by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"

    return "END"


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


def software_engineer(task: str) -> str:
    """Creates, modifies, and deletes code, manages files, runs shell commands, and collaborates with other agents."""
    final_state = graph.invoke(
        {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=task)]}
    )
    
    # Extract the final message content from the agent's internal state
    last_message_content = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_message = final_state["messages"][-1]
        
        if isinstance(last_message, (AIMessage, HumanMessage)):
            last_message_content = last_message.content
        elif isinstance(last_message, ToolMessage):
            last_message_content = str(last_message.content)
        elif isinstance(last_message, list):
            try:
                last_message_content = last_message[-1].content
            except (IndexError, AttributeError):
                last_message_content = str(last_message)
        else:
            last_message_content = str(last_message)
    
    # Handle case where content might be a list (for some LLM responses)
    if isinstance(last_message_content, list) and len(last_message_content) > 0:
        if isinstance(last_message_content[0], str):
            last_message_content = last_message_content[0]
        elif isinstance(last_message_content[0], dict) and last_message_content[0].get("type") == "text":
            last_message_content = str(last_message_content[0].get("text", ""))
        else:
            last_message_content = str(last_message_content[0])
    elif not isinstance(last_message_content, str):
        last_message_content = str(last_message_content)
    
    print(f"Software Engineer completed task. Final response: {last_message_content}")
    
    return last_message_content
