"""
Web Researcher Agent: Performs web searches and fetches page content.

Uses DuckDuckGo for searching and can fetch either raw HTML or rendered
content (via Selenium) depending on the tool used. Uses LangGraph for its
internal workflow.
"""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config

system_prompt = """You are web_researcher, a ReAct agent that can use the web to research answers.

You have a tool to search the web, and a tool to fetch the content of a web page.
```
"""
    
from tools.duck_duck_go_web_search import duck_duck_go_web_search
from tools.fetch_web_page_content import fetch_web_page_content

tools = [duck_duck_go_web_search, fetch_web_page_content]


def reasoning(state: MessagesState) -> dict:
    """
    Web Researcher's reasoning step. Invokes the LLM with its prompt and current state.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        dict: Update for the graph state with the LLM's response (AIMessage).
    """
    print("web_researcher is thinking...")
    messages = state['messages']
    tooled_up_model = config.default_langchain_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}


def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    """
    Checks the last message (AIMessage from reasoning) for tool calls.

    If tool calls are present, transitions to the 'tools' node.
    Otherwise, ends the graph execution for this agent.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        Literal["tools", END]: The next node to transition to.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("web_researcher thought this:")
            print(last_message.content)
        print()
        print("web_researcher is acting by invoking these tools:")
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
workflow.add_edge("tools", 'reasoning')

graph = workflow.compile()


def web_researcher(task: str) -> dict:
    """
    Executes the Web Researcher workflow to answer a query using web search.

    Takes a task/query, runs the internal LangGraph agent which uses search
    and fetch tools, and processes the final state to return a structured
    dictionary indicating the outcome and the research result.

    Args:
        task (str): The research query or task.

    Returns:
        dict: A dictionary containing:
              - 'status' (str): 'success' or 'failure'.
              - 'result' (str | None): The research findings on success, else None.
              - 'message' (str): The final message from the internal agent run,
                                 often the same as the result on success.
    """
    final_state = graph.invoke(
        {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=task)]}
    )

    # Extract the final message from the agent's internal state
    last_message_content = ""
    if final_state and 'messages' in final_state and final_state['messages']:
        # Check if the last message is an AIMessage or ToolMessage, handle accordingly
        last_message = final_state["messages"][-1]
        if isinstance(last_message, (AIMessage, HumanMessage)): # Could be AIMessage or HumanMessage if no tools called
             last_message_content = last_message.content
        elif isinstance(last_message, ToolMessage): # If the last step was a tool call
             last_message_content = str(last_message.content) # Content of the tool message
        elif isinstance(last_message, list):  # Handle LangGraph returning list sometimes?
            # Attempt to get content from the last item if it's a list of messages
            try:
                last_message_content = last_message[-1].content
            except (IndexError, AttributeError):
                 last_message_content = str(last_message) # Fallback
        else:
             last_message_content = str(last_message) # Fallback for other types


    # Determine status - assume success if the final message contains some research result.
    # TODO: Improve status detection (e.g., check for specific indicators of successful research vs. failure to find info).
    status = 'success' if last_message_content else 'failure'
    if "could not find" in last_message_content.lower() or "unable to research" in last_message_content.lower():
        status = 'failure'

    # Result is the research finding itself.
    result_data = last_message_content if status == 'success' else None

    print(f"Web Researcher finished task. Status: {status}. Final Message: {last_message_content}")

    return {
        "status": status,
        "result": result_data, 
        "message": last_message_content # Message can be the same as result for this agent
    }
