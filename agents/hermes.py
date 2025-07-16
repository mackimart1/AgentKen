"""
Hermes Agent: The central orchestrator for the AgentK system.

Handles user interaction, goal understanding, task planning, agent delegation
(using assign_agent_to_task), and result aggregation. Uses LangGraph to manage
its internal state machine and interaction flow.
"""

from typing import Literal, Sequence, TypedDict, Annotated, Optional, cast, Union  # Import Optional
import operator
import datetime

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk,
)
from langgraph.graph import StateGraph, END  # Removed MessagesState import
from langgraph.prebuilt import ToolNode

import utils
import config
import csv  # For logging training data
import os  # For checking file existence
import logging  # Use logging for info messages
from langchain_core.runnables.config import RunnableConfig

# OpenRouter uses standard HTTP errors, not Google-specific exceptions
from requests.exceptions import HTTPError, RequestException

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize memory manager (with error handling)
try:
    import memory_manager

    memory_manager_instance = memory_manager.MemoryManager()
    logger.info("Memory manager initialized successfully")
except ImportError as e:
    logger.warning(f"Could not import memory_manager: {e}")
    memory_manager_instance = None
except Exception as e:
    logger.warning(f"Could not initialize memory manager: {e}")
    memory_manager_instance = None

from tools.list_available_agents import list_available_agents
from tools.assign_agent_to_task import assign_agent_to_task

# from tools.predict_agent import predict_agent # Import the new prediction tool - commented out due to dependency issues

system_prompt = f"""You are Hermes, a ReAct agent that achieves goals for the user.

You are part of a system called AgentK - an autoagentic AGI.
AgentK is a self-evolving AGI made of agents that collaborate, and build new agents as needed, in order to complete tasks for a user.
Agent K is a modular, self-evolving AGI system that gradually builds its own mind as you challenge it to complete tasks.
The "K" stands kernel, meaning small core. The aim is for AgentK to be the minimum set of agents and tools necessary for it to bootstrap itself and then grow its own mind.

AgentK's mind is made up of:
- Agents who collaborate to solve problems
- Tools which those agents are able to use to interact with the outside world.

The agents that make up the kernel
- **hermes**: The orchestrator that interacts with humans to understand goals, manage the creation and assignment of tasks, and coordinate the activities of other agents.
- **agent_smith**: The architect responsible for creating and maintaining other agents. AgentSmith ensures agents are equipped with the necessary tools and tests their functionality.
- **tool_maker**: The developer of tools within the system, ToolMaker creates and refines the tools that agents need to perform their tasks, ensuring that the system remains flexible and well-equipped.
- **web_researcher**: The knowledge gatherer, WebResearcher performs in-depth online research to provide the system with up-to-date information, allowing agents to make informed decisions and execute tasks effectively.

You interact with a user in this specific order:
1. Reach a shared understanding on a goal.
2. Think of a detailed sequential plan for how to achieve the goal through the orchestration of agents. **Start your plan clearly with the marker `**Plan:**` on a new line.**
3. If a new kind of agent is required, assign a task to create that new kind of agent using the appropriate tool.
4. Assign agents and coordinate their activity based on your plan using the `assign_agent_to_task` tool or other relevant tools. **CRITICAL:** When you decide to use a tool in this step (or step 3), your response **MUST** contain both your textual statement of intent (e.g., "I will now assign the task...") AND the actual `tool_calls` data for the system to execute. Do not separate these into different messages.
5. Respond to the user with the final result once the goal is achieved, or if you encounter an unrecoverable error, or if you genuinely need their input to proceed (but not just for confirming a tool call).

Further guidance:
- You have tools: `list_available_agents`, `assign_agent_to_task`, `predict_agent`.
- **Agent Assignment Workflow:**
    1. **Plan:** Decide which agent seems appropriate for a sub-task.
    2. **Predict (Optional but Recommended):** If unsure or for confirmation, use `predict_agent` with the sub-task description. This returns agent probabilities based on a trained model.
    3. **Decide:** Use your reasoning, the agent list (if needed), and the prediction probabilities to choose the final agent.
    4. **Assign:** Use `assign_agent_to_task` with the chosen agent and the sub-task description.
    - **Tool Results (`assign_agent_to_task`):**
    - The tool returns a dictionary: {{'status': 'success'|'failure', 'result': ..., 'message': ...}}.
    - `result`: Contains the primary output or result from the agent, if applicable (can be any data type).
    - `message`: A string providing details about the outcome, context, or error information.
- **Crucially, you must check the `status` field in the tool's response.**
    - If `status` is 'success', use the `result` and `message` to inform your next step in the plan.
    - If `status` is 'failure', analyze the `message` to understand the error, adapt your plan if possible (e.g., try a different agent, modify the task, ask the user for help), or report the failure to the user if you cannot proceed.
- Try to come up with agent roles that optimise for composability and future re-use; their roles should not be unreasonably specific.
- Use the `list_available_agents` tool *only if needed* during planning to see available agents and their descriptions. Do not invoke it unnecessarily.
- **CRITICAL INSTRUCTION REITERATED:** When you decide to use a tool as part of your plan (steps 3 or 4 above), your single response message *must* include both your plain text statement of intent AND the `tool_calls` data. Failure to include `tool_calls` in the same message will prevent the tool from running automatically. Also, **NEVER** include raw code like `print(tool_name(...))` in your text response.
"""

# Load tools using the refactored manifest-based function
# Note: predict_agent might not be loaded if its models are missing,
# but all_tool_functions handles this. Hermes should check tool availability if needed.
tools = utils.all_tool_functions()
# Safely get tool names for logging
tool_names_for_log = []
for t in tools:
    if hasattr(t, "name"):
        tool_names_for_log.append(t.name)
    else:
        # Log the type if name is missing, might indicate an issue
        logger.warning(
            f"Tool object of type {type(t)} is missing '.name' attribute during Hermes init."
        )
        tool_names_for_log.append(f"[Unknown Tool Type: {type(t).__name__}]")
logger.info(f"Hermes initialized with tools: {tool_names_for_log}")


# --- Define Custom State ---
class HermesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan_step_count: int
    initial_goal: Optional[str]  # Use Optional[str] for compatibility


def feedback_and_wait_on_human_input(state: HermesState) -> dict:
    """
    Handles interaction with the user.

    Prints the last message content to the console and waits for user input.
    If it's the start of the conversation, prompts the user.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        dict: Update for the graph state with the new HumanMessage, reset step count, and potentially the initial goal.
    """
    # Determine message to show user
    if len(state["messages"]) == 1:
        message_to_human = "What can I help you with?"
    else:
        # Show the last AI message content, handling potential list structure
        last_ai_content = state["messages"][-1].content
        if isinstance(last_ai_content, list) and len(last_ai_content) > 0:
            # If content is a list (like from Gemini function calling), print only the first part (text)
            message_to_human = (
                last_ai_content[0]
                if isinstance(last_ai_content[0], str)
                else str(last_ai_content[0])
            )
        elif isinstance(last_ai_content, str):
            # If it's just a string, use it directly
            message_to_human = last_ai_content
        else:
            # Fallback for unexpected content types
            message_to_human = str(last_ai_content)
    print(message_to_human)

    # Get user input
    human_input = ""
    while not human_input.strip():
        human_input = input("> ")

    # Prepare state update
    update_dict = {
        "messages": [HumanMessage(content=human_input)],
        "plan_step_count": 0,  # Reset step count on new user input
    }

    # Capture initial goal if not already set
    if not state.get("initial_goal"):
        update_dict["initial_goal"] = human_input
        print(f"(Initial goal captured: '{human_input[:50]}...')")

    return update_dict


def check_for_exit(
    state: HermesState,
) -> Literal["reasoning", END]:  # Updated state type
    """
    Checks the last message from the user to see if they typed 'exit'.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        Literal["reasoning", END]: The next node to transition to.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        # Handle content that might be a list or string
        content = last_message.content
        if isinstance(content, list):
            # If content is a list, extract text from first element
            content_text = ""
            if content and isinstance(content[0], str):
                content_text = content[0]
            elif content and isinstance(content[0], dict) and content[0].get("type") == "text":
                content_text = content[0].get("text", "")
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = str(content)
            
        if content_text.lower() == "exit":
            return END
    
    return "reasoning"


def reasoning(state: HermesState) -> dict:
    """
    The core reasoning step. Checks for early stopping, then memory shortcuts,
    then invokes the LLM with the current message history and tools.

    Args:
        state (HermesState): The current graph state.

    Returns:
        dict: Update for the graph state with the LLM's response (AIMessage).
    """
    print()
    print("hermes is thinking...")
    current_messages, plan_step_count, initial_goal = get_state_values(
        state
    )  # Use helper

    # Add acknowledgment detection at the start
    if current_messages and isinstance(current_messages[-1], HumanMessage):
        # Handle content that might be a list or string
        content = current_messages[-1].content
        if isinstance(content, list):
            # If content is a list, extract text from first element
            last_human_msg = ""
            if content and isinstance(content[0], str):
                last_human_msg = content[0]
            elif content and isinstance(content[0], dict) and content[0].get("type") == "text":
                last_human_msg = content[0].get("text", "")
            else:
                last_human_msg = str(content[0]) if content else ""
            last_human_msg = last_human_msg.lower().strip()
        elif isinstance(content, str):
            last_human_msg = content.lower().strip()
        else:
            last_human_msg = str(content).lower().strip()
            
        # Check if this is a simple acknowledgment
        acknowledgments = {
            "ok",
            "okay",
            "alright",
            "got it",
            "understood",
            "proceed",
            "continue",
            "great",
        }
        if last_human_msg in acknowledgments:
            # Check if we're in the middle of a task
            for msg in reversed(current_messages[:-1]):
                if isinstance(msg, AIMessage):
                    msg_content = msg.content
                    if isinstance(msg_content, str) and "waiting" in msg_content.lower():
                        # Skip full reasoning for acknowledgments during waiting periods
                        return {
                            "messages": [
                                AIMessage(content="Continuing with the current task...")
                            ]
                        }
                break

    # --- Early Stopping Check ---
    EARLY_STOP_THRESHOLD = 3  # Define the threshold
    if plan_step_count >= EARLY_STOP_THRESHOLD:
        print(
            f"Plan step count ({plan_step_count}) reached threshold ({EARLY_STOP_THRESHOLD}). Evaluating plan viability..."
        )
        if not initial_goal:
            print(
                "Warning: Initial goal not found in state for early stopping evaluation."
            )
            # Proceed without evaluation if goal is missing
        else:
            # Prepare limited history for evaluation prompt
            recent_history_str = "\n".join(
                [
                    f"{type(m).__name__}: {m.content[:200]}..."
                    for m in current_messages[-6:]
                ]  # Last ~3 turns
            )
            evaluation_prompt = f"""Analyze the likelihood of success for the current plan based on the initial goal and recent interactions.
Initial Goal: {initial_goal}
Recent History Summary:
{recent_history_str}

Based ONLY on the provided goal and history summary, is the current plan likely to succeed? Consider recent tool failures or lack of progress.
Respond ONLY with 'PROCEED' or 'STOP: [Brief reason, e.g., repeated tool failures, agent unable to complete subtask]'."""

            evaluation_successful = False
            while not evaluation_successful:  # Loop for quota handling
                try:
                    if config.default_langchain_model is None:
                        raise ValueError(
                            "Default language model not initialized for evaluation."
                        )

                    evaluation_response = config.default_langchain_model.invoke(
                        [SystemMessage(content=evaluation_prompt)]
                    )
                    # Fix: Only call .strip() on a string, handle list case
                    evaluation_content = evaluation_response.content
                    if isinstance(evaluation_content, str):
                        evaluation_text = evaluation_content.strip()
                    elif isinstance(evaluation_content, list) and len(evaluation_content) > 0:
                        if isinstance(evaluation_content[0], str):
                            evaluation_text = evaluation_content[0].strip()
                        elif isinstance(evaluation_content[0], dict) and evaluation_content[0].get("type") == "text":
                            evaluation_text = str(evaluation_content[0].get("text", "")).strip()
                        else:
                            evaluation_text = str(evaluation_content[0]).strip()
                    else:
                        evaluation_text = str(evaluation_content).strip()
                    evaluation_successful = True  # Mark success

                    if isinstance(evaluation_text, str) and evaluation_text.startswith("STOP:"):
                        reason = evaluation_text.replace("STOP:", "").strip()
                        print(f"Evaluation result: STOP. Reason: {reason}")
                        stop_message = f"Stopping task execution early after {plan_step_count} steps: {reason}. Returning control."
                        # Return message to be sent back to user via feedback node
                        return {"messages": [AIMessage(content=stop_message)]}
                    elif evaluation_text == "PROCEED":
                        print("Evaluation result: PROCEED.")
                        # Continue to normal reasoning flow
                    else:
                        print(
                            f"Unexpected response from evaluation check: {evaluation_text}. Proceeding."
                        )
                        # Continue to normal reasoning flow

                except (HTTPError, RequestException) as eval_e:
                    print("\n--------------------------------------------------")
                    print(
                        f"OpenRouter API Error during early stopping evaluation for Hermes: {eval_e}"
                    )
                    print("Please enter a new OpenRouter API Key to continue:")
                    new_key_eval = input("> ").strip()
                    if not new_key_eval:
                        print("No key entered. Cannot evaluate plan.")
                        break
                    try:
                        config.reinitialize_openrouter_model(new_key_eval)
                        print("Model reinitialized. Retrying evaluation...")
                    except Exception as config_eval_e:
                        print(
                            f"Failed to reinitialize OpenRouter model: {config_eval_e}"
                        )
                        break
                except Exception as eval_e_other:
                    logger.error(
                        f"An unexpected error occurred during Hermes early stopping evaluation: {eval_e_other}",
                        exc_info=True,
                    )
                    print(
                        "Error during evaluation check. Proceeding with main reasoning."
                    )
                    evaluation_successful = True  # Break loop on unexpected error

    # --- End Early Stopping Check ---

    # --- Extract current task/query ---
    query_context = "Initial state"  # Default
    task_description = "General context"  # Default
    if current_messages:
        # Find the most recent human message for the task context
        for msg in reversed(current_messages):
            if isinstance(msg, HumanMessage):
                query_context = msg.content
                task_description = msg.content
                break
        else:  # If no human message, use the last message content
            query_context = (
                current_messages[-1].content
                if current_messages[-1].content
                else "Previous action context"
            )
            task_description = query_context  # Approximate task

    # --- Memory Retrieval ---
    print(f"Retrieving memories for task: {task_description[:100]}...")
    # Retrieve memories, sorting by importance by default
    relevant_memories = []
    if memory_manager_instance is not None:
        try:
            relevant_memories = memory_manager_instance.retrieve_memories(
                memory_type="",  # Use empty string instead of None
                agent_name="",  # Use empty string instead of None
                min_importance=5,  # Get memories with importance >= 5
                limit=5,  # Get up to 5 memories
            )
        except Exception as mem_e:
            logger.warning(f"Failed to retrieve memories: {mem_e}")
            relevant_memories = []
    else:
        print("Memory manager not available, skipping memory retrieval.")

    # --- Shortcut Check ---
    shortcut_found = False
    shortcut_response_message = None

    if relevant_memories:
        print("Checking memory for shortcuts...")
        # Format memories properly by extracting their values
        formatted_memories = "\n".join(
            [f"- {mem['value']}" for mem in relevant_memories]
        )
        shortcut_prompt = f"""Analyze the following task and retrieved memories.
Task: {task_description}
Memories:
{formatted_memories}

Based ONLY on the provided memories, is there a direct pre-existing solution, a specific existing tool, or a specific existing agent suitable for this task that would allow skipping the main reasoning process?
Respond ONLY with:
- "SHORTCUT: [Explanation of shortcut and proposed action/tool/agent, e.g., 'Use tool X', 'Assign to agent Y', 'Found solution: ...']" if a clear shortcut exists in the memories.
- "NO SHORTCUT" if no clear shortcut is found in the memories."""

        shortcut_check_successful = False
        while (
            not shortcut_check_successful
        ):  # Loop for quota handling on shortcut check
            try:
                # Use the base model for this simple check (no tools needed)
                # Ensure config.default_langchain_model is available and initialized
                if config.default_langchain_model is None:
                    raise ValueError(
                        "Default language model not initialized in config."
                    )

                # Ensure the prompt is not empty before invoking
                if not shortcut_prompt.strip():
                    print("Shortcut prompt is empty, skipping shortcut check.")
                    shortcut_text = (
                        "NO SHORTCUT"  # Treat as no shortcut if prompt is empty
                    )
                else:
                    # Create a proper message chain for Gemini
                    messages = [
                        SystemMessage(
                            content="You are a helpful assistant that analyzes tasks and memories to find shortcuts."
                        ),
                        HumanMessage(content=shortcut_prompt),
                    ]
                    shortcut_check_response = config.default_langchain_model.invoke(
                        messages
                    )
                    shortcut_content = shortcut_check_response.content
                    if isinstance(shortcut_content, str):
                        shortcut_text = shortcut_content.strip()
                    elif isinstance(shortcut_content, list) and len(shortcut_content) > 0:
                        if isinstance(shortcut_content[0], str):
                            shortcut_text = shortcut_content[0].strip()
                        elif isinstance(shortcut_content[0], dict) and shortcut_content[0].get("type") == "text":
                            shortcut_text = str(shortcut_content[0].get("text", "")).strip()
                        else:
                            shortcut_text = str(shortcut_content[0]).strip()
                    else:
                        shortcut_text = str(shortcut_content).strip()

                shortcut_check_successful = (
                    True  # Mark as successful if invoke didn't raise or was skipped
                )

                if isinstance(shortcut_text, str) and shortcut_text.startswith("SHORTCUT:"):
                    explanation = shortcut_text.replace("SHORTCUT:", "").strip()
                    print(f"Memory shortcut found: {explanation}")
                    # Prepare AIMessage containing the shortcut explanation
                    shortcut_response_message = AIMessage(
                        content=explanation, tool_calls=[]
                    )  # Ensure no tool calls for shortcut message
                    shortcut_found = True
                elif shortcut_text == "NO SHORTCUT":
                    print("No memory shortcut found.")
                else:
                    print(
                        f"Unexpected response from shortcut check: {shortcut_text}. Proceeding with main reasoning."
                    )

            except (HTTPError, RequestException) as e:
                print("\n--------------------------------------------------")
                print(f"OpenRouter API Error during shortcut check for Hermes: {e}")
                print("Please enter a new OpenRouter API Key to continue:")
                new_key = input("> ").strip()
                if not new_key:
                    print("No key entered. Cannot proceed.")
                    raise ValueError(
                        "User did not provide a new API key after quota error."
                    ) from e
                print("Attempting to reinitialize model with the new key...")
                try:
                    config.reinitialize_openrouter_model(new_key)
                    print("Model reinitialized. Retrying shortcut check...")
                except Exception as config_e:
                    print(f"Failed to reinitialize OpenRouter model: {config_e}")
                    raise config_e
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during Hermes shortcut check: {e}",
                    exc_info=True,
                )
                print("Error during shortcut check. Proceeding with main reasoning.")
                shortcut_check_successful = True  # Break loop on unexpected error

    # --- Return Shortcut or Proceed to Main Reasoning ---
    if shortcut_found and shortcut_response_message:
        # Return the shortcut explanation directly as an AIMessage
        # Hermes will process this in the next cycle (e.g., call the suggested tool/agent)
        return {"messages": [shortcut_response_message]}

    # --- Main Reasoning (if no shortcut found) ---
    print("Proceeding with main reasoning...")
    # Prepare messages for LLM, including retrieved context
    messages_for_llm = list(current_messages)
    if relevant_memories:  # Add context if memories were retrieved, even if no shortcut
        context_header = (
            "Relevant context from past interactions (no direct shortcut found):\n"
        )
        formatted_memories = "\n".join([f"- {mem}" for mem in relevant_memories])
        found_system_message = False
        for i, msg in enumerate(messages_for_llm):
            if isinstance(msg, SystemMessage):
                original_content = msg.content
                # Ensure system_prompt (global) is used if it's the initial message
                if i == 0 and msg.content == system_prompt:
                    original_content = system_prompt
                # Avoid double-prepending context if reasoning loops
                if not (isinstance(original_content, str) and original_content.startswith(
                    "Relevant context from past interactions"
                )):
                    messages_for_llm[i] = SystemMessage(
                        content=f"{context_header}{formatted_memories}\n\n---\n\n{original_content}"
                    )
                found_system_message = True
                break
        if not found_system_message:
            messages_for_llm.insert(
                0, SystemMessage(content=f"{context_header}{formatted_memories}")
            )

    # --- Sanitize messages before sending to LLM ---
    sanitized_messages_for_llm = sanitize_messages(messages_for_llm)
    # --- End Sanitization ---

    # --- Remove Old/Flawed Plan Memory Logic ---
    # The previous logic checked the wrong message and was unreliable.
    # We will add new logic after the response is received.

    # Main LLM call with retry logic
    while True:
        try:
            if config.default_langchain_model is None:
                raise ValueError(
                    "Default language model not initialized in config before main reasoning."
                )

            # Ensure messages_for_llm is not empty before invoking
            if not messages_for_llm:
                logging.error(
                    "Hermes main reasoning: messages_for_llm is empty. Cannot invoke LLM."
                )
                # Return a generic error message or handle appropriately
                return {
                    "messages": [
                        AIMessage(
                            content="Internal error: Cannot proceed with empty message list."
                        )
                    ]
                }

            # Use Google Gemini for tool calling from hybrid configuration
            tool_model = config.get_model_for_tools()
            if tool_model is None:
                # Fallback to default model if hybrid setup fails
                tool_model = config.default_langchain_model
                logger.warning("Using fallback model for tools - may not support function calling")
            
            tooled_up_model = tool_model.bind_tools(tools)
            # Use the sanitized messages list for the invocation
            response = tooled_up_model.invoke(
                sanitized_messages_for_llm
            )  # Get the main response

            # --- Clean the response content before further processing ---
            cleaned_content = response.content
            if isinstance(response.content, list) and len(response.content) > 0:
                # Assume the first element is the desired text if content is a list
                cleaned_content = (
                    response.content[0]
                    if isinstance(response.content[0], str)
                    else str(response.content[0])
                )

            # Explicitly remove ```tool_code blocks if they appear in the text content
            if isinstance(cleaned_content, str) and "```tool_code" in cleaned_content:
                # Attempt to remove the block - this might be brittle
                parts = cleaned_content.split("```tool_code", 1)
                cleaned_content = parts[
                    0
                ].strip()  # Keep only the part before the block

            # Create a new AIMessage with cleaned content but original tool_calls
            # Ensure all relevant attributes are copied
            ai_response = cast(AIMessage, response)
            
            # Fix tool calls to ensure they have proper IDs
            fixed_tool_calls = []
            if ai_response.tool_calls:
                import uuid as uuid_module
                for i, tool_call in enumerate(ai_response.tool_calls):
                    # Ensure tool_call is a dict and has an 'id' field
                    if isinstance(tool_call, dict):
                        if not tool_call.get("id"):
                            tool_call["id"] = f"call_{uuid_module.uuid4().hex[:8]}"
                        fixed_tool_calls.append(tool_call)
                    else:
                        # If tool_call is not a dict, try to convert it or create a proper structure
                        fixed_tool_call = {
                            "id": f"call_{uuid_module.uuid4().hex[:8]}",
                            "name": getattr(tool_call, "name", f"unknown_tool_{i}"),
                            "args": getattr(tool_call, "args", {})
                        }
                        fixed_tool_calls.append(fixed_tool_call)
            
            cleaned_response = AIMessage(
                content=cleaned_content,
                tool_calls=fixed_tool_calls,
                id=getattr(ai_response, "id", None),
                additional_kwargs=getattr(ai_response, "additional_kwargs", {}),
                response_metadata=getattr(ai_response, "response_metadata", {}),
                name=getattr(ai_response, "name", None),
                tool_call_chunks=getattr(
                    ai_response, "tool_call_chunks", None
                ),  # Preserve if exists
            )
            # --- End Cleaning ---

            # --- Plan Compression & Memory Storage (using cleaned_content) ---
            compressed_summary = None
            # Use cleaned_content for plan detection/compression
            if (
                cleaned_content
                and isinstance(cleaned_content, str)
                and "**Plan:**" in cleaned_content
            ):
                print("Plan detected in response. Attempting compression...")
                try:
                    # Extract the detailed plan text from cleaned_content
                    detailed_plan_text = cleaned_content.split("**Plan:**", 1)[
                        1
                    ].strip()

                    # Check if the extracted plan text is actually present
                    if detailed_plan_text:
                        # Define compression prompt
                        compression_prompt = f"""Compress the following detailed plan into a concise action summary (1-2 sentences) suitable for memory storage. Focus on the overall goal and key actions/stages, omitting step-by-step details.

Detailed Plan:
{detailed_plan_text}

Compressed Summary:"""

                        # Nested loop for compression call quota handling
                        compression_successful = False
                        while not compression_successful:
                            # CORRECTED INDENTATION BLOCK START
                            try:  # Line ~356
                                if config.default_langchain_model is None:
                                    raise ValueError(
                                        "Default language model not initialized before compression."
                                    )

                                # Ensure compression_prompt is not empty
                                if not compression_prompt.strip():
                                    print(
                                        "Compression prompt is empty, skipping compression."
                                    )
                                    compressed_summary = (
                                        None  # Ensure summary is None if skipped
                                    )
                                    compression_successful = (
                                        True  # Mark as 'successful' to exit loop
                                    )
                                else:
                                    # Use HumanMessage instead of SystemMessage for the compression prompt
                                    compression_response = (
                                        config.default_langchain_model.invoke(
                                            [HumanMessage(content=compression_prompt)]
                                        )
                                    )
                                    compression_content = compression_response.content
                                    if isinstance(compression_content, str):
                                        compressed_summary = compression_content.strip()
                                    elif isinstance(compression_content, list) and len(compression_content) > 0:
                                        if isinstance(compression_content[0], str):
                                            compressed_summary = compression_content[0].strip()
                                        elif isinstance(compression_content[0], dict) and compression_content[0].get("type") == "text":
                                            compressed_summary = str(compression_content[0].get("text", "")).strip()
                                        else:
                                            compressed_summary = str(compression_content[0]).strip()
                                    else:
                                        compressed_summary = str(compression_content).strip()
                                    compression_successful = True  # Mark success
                                    print(
                                        f"Plan compressed successfully: {compressed_summary}"
                                    )

                            except (HTTPError, RequestException) as comp_e:
                                print(
                                    "\n--------------------------------------------------"
                                )
                                print(
                                    f"OpenRouter API Error during plan compression for Hermes: {comp_e}"
                                )
                                print(
                                    "Please enter a new OpenRouter API Key to continue:"
                                )
                                new_key_comp = input("> ").strip()
                                if not new_key_comp:
                                    print("No key entered. Cannot compress plan.")
                                    break  # Exit compression loop if no key provided
                                print(
                                    "Attempting to reinitialize model with the new key..."
                                )
                                try:
                                    config.reinitialize_openrouter_model(new_key_comp)
                                    print(
                                        "Model reinitialized. Retrying compression call..."
                                    )
                                except Exception as config_comp_e:
                                    print(
                                        f"Failed to reinitialize OpenRouter model during compression: {config_comp_e}"
                                    )
                                    break  # Exit compression loop if re-init fails
                            except Exception as comp_e_other:
                                logger.error(
                                    f"An unexpected error occurred during Hermes plan compression: {comp_e_other}",
                                    exc_info=True,
                                )
                                print(
                                    "Error during plan compression. Skipping summary storage."
                                )
                                compression_successful = (
                                    True  # Break loop on unexpected error
                                )
                            # CORRECTED INDENTATION BLOCK END

                        # Store compressed summary if successful
                        if compressed_summary and memory_manager_instance is not None:
                            try:
                                memory_manager_instance.add_memory(
                                    key=f"plan_summary_{datetime.datetime.now().timestamp()}",  # Generate unique key
                                    value=compressed_summary,
                                    memory_type="plan_summary",
                                    agent_name="hermes",
                                )
                            except Exception as mem_add_e:
                                logger.warning(f"Failed to add plan summary to memory: {mem_add_e}")
                        else:
                            # If detailed_plan_text was empty, skip compression
                            print(
                                "Skipping plan compression as extracted plan text was empty."
                            )

                # Unindent this except block to align with the outer try (around line 348)
                except Exception as outer_comp_e:
                    # Catch errors in extracting plan or setting up compression
                    logger.error(
                        f"Error setting up plan compression: {outer_comp_e}",
                        exc_info=True,
                    )
                    print(
                        "Error during plan compression setup. Skipping summary storage."
                    )

            # Add general reasoning output to memory (using cleaned_content)
            # Avoid storing the full plan again if we stored the summary
            if (
                cleaned_content
                and (not hasattr(cleaned_response, "tool_calls") or not cleaned_response.tool_calls)
                and not compressed_summary
                and memory_manager_instance is not None
            ):
                # Store if it's content AND not a tool call AND we didn't store a summary
                try:
                    # Ensure cleaned_content is a string before passing to add_memory
                    memory_value = None
                    if isinstance(cleaned_content, str):
                        memory_value = cleaned_content
                    elif isinstance(cleaned_content, list) and len(cleaned_content) > 0:
                        if isinstance(cleaned_content[0], str):
                            memory_value = cleaned_content[0]
                        elif isinstance(cleaned_content[0], dict) and cleaned_content[0].get("type") == "text":
                            memory_value = str(cleaned_content[0].get("text", ""))
                        else:
                            memory_value = str(cleaned_content[0])
                    else:
                        memory_value = str(cleaned_content)
                    memory_manager_instance.add_memory(
                        key=f"reasoning_{datetime.datetime.now().timestamp()}",  # Generate unique key
                        value=memory_value,
                        memory_type="reasoning_output",
                        agent_name="hermes",
                    )
                except Exception as mem_add_e:
                    logger.warning(f"Failed to add reasoning output to memory: {mem_add_e}")

            # --- End Plan Compression & Memory Storage ---

            # Return the CLEANED response
            return {"messages": [cleaned_response]}

        except (HTTPError, RequestException) as e:
            print("\n--------------------------------------------------")
            print(f"OpenRouter API Error during main reasoning for Hermes: {e}")
            print("Please enter a new OpenRouter API Key to continue:")
            new_key = input("> ").strip()
            if not new_key:
                print("No key entered. Cannot proceed.")
                raise ValueError(
                    "User did not provide a new API key after quota error."
                ) from e
            print("Attempting to reinitialize model with the new key...")
            try:
                config.reinitialize_openrouter_model(new_key)
                print("Model reinitialized. Retrying API call...")
            except Exception as config_e:
                print(f"Failed to reinitialize OpenRouter model: {config_e}")
                raise config_e

        except Exception as e:
            logging.error(
                f"An unexpected error occurred during Hermes main model invocation: {e}",
                exc_info=True,
            )
            raise e


def check_for_tool_calls(
    state: HermesState,
) -> Literal["tools", "feedback_and_wait_on_human_input"]:
    """
    Checks the last message (AIMessage from reasoning) for tool calls or early stopping.

    Args:
        state (HermesState): The current graph state.

    Returns:
        Literal["tools", "feedback_and_wait_on_human_input"]: The next node.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message is an early stopping message
    # REVISED: Use try-except for robustness against AttributeError on .startswith()
    is_stopping_message = False
    if isinstance(last_message, AIMessage):
        try:
            # Only attempt startswith if content is a string
            if isinstance(
                last_message.content, str
            ) and last_message.content.startswith("Stopping task execution early:"):
                is_stopping_message = True
        except AttributeError:
            # If content is not a string or lacks startswith (e.g., list), it's not the stopping message.
            pass  # is_stopping_message remains False

    if is_stopping_message:
        return "feedback_and_wait_on_human_input"  # Go back to user

    # Original tool call check
    # Only AIMessage has tool_calls attribute
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # if not last_message.content.strip() == "": # Commented out to hide tool_code print
        #     print("hermes thought this:")
        #     print(last_message.content)
        # print() # Commented out
        # print("hermes is acting by invoking these tools:") # Commented out
        # print([tool_call["name"] for tool_call in last_message.tool_calls]) # Commented out
        return "tools"
    else:
        return "feedback_and_wait_on_human_input"


acting = ToolNode(tools)

workflow = StateGraph(HermesState)  # Use the custom state
workflow.add_node("feedback_and_wait_on_human_input", feedback_and_wait_on_human_input)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("feedback_and_wait_on_human_input")
workflow.add_conditional_edges(
    "feedback_and_wait_on_human_input",
    check_for_exit,
)
workflow.add_conditional_edges(
    "reasoning",
    check_for_tool_calls,
)
# workflow.add_edge("tools", 'reasoning') # Replace simple edge with conditional


# --- Condition to check the outcome of the tool call node ---
def check_tool_outcome_state_update(state: HermesState) -> Optional[dict]:
    """
    State update function for add_conditional_edges: returns dict if step count should increment, else None.
    """
    # The ToolNode adds ToolMessages to the state. The most recent one contains the output.
    last_message = state["messages"][-1]
    current_step_count = state.get("plan_step_count", 0)
    if (
        isinstance(last_message, ToolMessage)
        and isinstance(last_message.content, dict)
        and "status" in last_message.content
    ):
        tool_status = last_message.content.get("status")
        for i in range(len(state["messages"]) - 2, -1, -1):
            prev_msg = state["messages"][i]
            if (isinstance(prev_msg, AIMessage) or isinstance(prev_msg, AIMessageChunk)):
                ai_msg = cast(AIMessage, prev_msg)
                if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        if tool_call.get("name") == "assign_agent_to_task":
                            if tool_status == "success":
                                return {"plan_step_count": current_step_count + 1}
                break
    return None

workflow.add_conditional_edges(
    "tools",
    lambda state: "reasoning"
)


graph = workflow.compile(checkpointer=utils.checkpointer)


# --- Message Sanitization Helper ---
def sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """
    Iterates through messages and removes unsupported parts like 'executable_code'.

    Args:
        messages: The original list of messages.

    Returns:
        A new list of messages with unsupported parts filtered out.
    """
    sanitized_messages = []
    supported_types = {"text", "image_url", "media"}  # Define supported types

    for msg in messages:
        if isinstance(msg.content, str):
            # If content is just a string, it's fine
            sanitized_messages.append(msg)
        elif isinstance(msg.content, list):
            # If content is a list of parts, filter them
            filtered_content = []
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in supported_types:
                    filtered_content.append(part)
                elif isinstance(part, str):  # Sometimes parts can be just strings
                    filtered_content.append({"type": "text", "text": part})

            # Only add message back if it still has content after filtering
            if filtered_content:
                # Create a new message of the same type with filtered content
                # Handle different message types that might have list content
                if isinstance(msg, HumanMessage):
                    new_msg = HumanMessage(content=filtered_content)
                elif isinstance(msg, AIMessage):
                    # Preserve tool_calls if they exist
                    new_msg = AIMessage(
                        content=filtered_content,
                        tool_calls=getattr(msg, "tool_calls", []),
                    )
                elif isinstance(
                    msg, AIMessageChunk
                ):  # Handle potential chunks if streaming
                    new_msg = AIMessageChunk(
                        content=filtered_content,
                        tool_calls=getattr(msg, "tool_calls", []),
                    )
                # Add other message types here if necessary (e.g., SystemMessage, ToolMessage usually have string content)
                else:
                    # Fallback: Keep original if type handling not specified, or skip if unsafe
                    # For safety, let's skip if we don't know how to reconstruct
                    logger.warning(
                        f"Skipping message of type {type(msg)} during sanitization due to list content."
                    )
                    continue  # Skip adding this message

                # Copy metadata if present
                if hasattr(msg, "additional_kwargs"):
                    new_msg.additional_kwargs = msg.additional_kwargs
                if hasattr(msg, "response_metadata"):
                    new_msg.response_metadata = msg.response_metadata
                if hasattr(msg, "id"):
                    new_msg.id = msg.id
                if hasattr(msg, "name"):
                    new_msg.name = msg.name

                sanitized_messages.append(new_msg)
            else:
                logger.warning(
                    f"Message content became empty after filtering unsupported parts: {msg}"
                )
        else:
            # Keep message if content type is neither string nor list (shouldn't happen often)
            sanitized_messages.append(msg)

    return sanitized_messages


# --- End Message Sanitization Helper ---


# Helper to get current state values safely
def get_state_values(state: HermesState) -> tuple:
    messages = state.get("messages", [])
    plan_step_count = state.get("plan_step_count", 0)
    initial_goal = state.get("initial_goal", None)
    return messages, plan_step_count, initial_goal


# Ensure this function definition starts at the top level (no indent)
def hermes(uuid: str):
    """
    Runs the main Hermes agent loop.

    Initializes the LangGraph agent with the system prompt and manages the
    interaction flow based on the compiled graph.

    Args:
        uuid (str): A unique identifier for the conversation thread.
    """
    print(f"Starting session with AgentK (id:{uuid})")
    print("Type 'exit' to end the session.")

    # Initialize with default values for the custom state
    initial_state = {
        "messages": [SystemMessage(content=system_prompt)],
        "plan_step_count": 0,
        "initial_goal": None,
    }
    # Ensure this return is correctly indented within the function
    # Increase recursion limit to handle potentially longer plans
    config_with_limit = {"configurable": {"thread_id": uuid}, "recursion_limit": 66}
    config_with_limit = cast(RunnableConfig, config_with_limit)
    return graph.invoke(initial_state, config=config_with_limit)
