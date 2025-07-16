"""
Tool Maker Agent: The tool developer for the AgentK system.

Responsible for creating new LangChain tools based on task requirements.
It writes the tool code and tests, formats/lints the code, runs the tests,
and reports the outcome back to Hermes. Uses LangGraph for its internal workflow.
"""

from typing import Literal, Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# OpenRouter uses standard HTTP errors, not Google-specific exceptions
from requests.exceptions import HTTPError, RequestException
import logging
import sys
import os
import time
import re

# --- Explicitly add project root to sys.path ---
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import modules with error handling
try:
    import memory_manager

    memory_manager_instance = memory_manager.MemoryManager()
    MEMORY_AVAILABLE = True
except ImportError:
    print("Warning: memory_manager not available")
    memory_manager = None
    MEMORY_AVAILABLE = False

import utils
import config

# Setup logger for tool_maker
logger = logging.getLogger(__name__)

system_prompt = """You are tool_maker, a ReAct agent that develops LangChain tools for other agents in the AgentK system.

Your goal is to create a functional and tested tool based on the user's request.

**Workflow:**
1.  **Analyze Request:** Understand the tool requirements and determine the appropriate tool name and functionality.
2.  **Check for Existing Tools:** Before creating new code, check if a similar tool already exists in the tools directory.
3.  **Write Code & Test:** Create `tools/tool_name.py` and `tests/tools/test_tool_name.py`. Include comprehensive docstrings and error handling.
4.  **Format & Lint:** Run black formatter and ruff linter to ensure code quality.
5.  **Test:** Run unit tests to verify functionality.
6.  **Validate:** Ensure the tool can be imported and used correctly.
7.  **Report:** Provide clear status on success/failure with detailed information.

**Key Rules:**
- Tool files go in `tools/`. Test files go in `tests/tools/`.
- Tool filename and function name MUST match. One tool function per file.
- Use the `@tool` decorator from `langchain_core.tools`.
- Include comprehensive docstrings with parameter descriptions and return types.
- Add proper error handling and input validation.
- Use type hints for better code quality.
- Test edge cases and error conditions.
- Follow Python best practices (PEP 8, etc.).

**Dependencies:**
- Manage Python dependencies via `requirements.txt` and `pip install -r requirements.txt`.
- Manage OS dependencies (Debian 11) via `apt-packages-list.txt`.
- Use `request_human_input` if external input (like API keys) is needed.

**Response Style:** 
- Use inner monologue for planning and debugging thoughts before tool calls.
- Final response should clearly indicate success/failure and provide the tool name.
- Be specific about any issues encountered and how they were resolved.

**Example Tool Structure:**
```python
# tools/get_smiley.py
from langchain_core.tools import tool
from typing import Optional

@tool
def get_smiley(mood: str = "happy") -> str:
    \"\"\"Get a smiley based on mood.
    
    Args:
        mood (str): The mood for the smiley. Options: happy, sad, neutral.
        
    Returns:
        str: The appropriate smiley emoji.
        
    Raises:
        ValueError: If mood is not recognized.
    \"\"\"
    mood_map = {
        "happy": ":)",
        "sad": ":(",
        "neutral": ":|"
    }
    
    if mood not in mood_map:
        raise ValueError(f"Unknown mood: {mood}. Available: {list(mood_map.keys())}")
    
    return mood_map[mood]
```

```python
# tests/tools/test_get_smiley.py
import unittest
from tools.get_smiley import get_smiley

class TestGetSmiley(unittest.TestCase):
    def test_happy_smiley(self):
        result = get_smiley.invoke({"mood": "happy"})
        self.assertEqual(result, ":)")
    
    def test_sad_smiley(self):
        result = get_smiley.invoke({"mood": "sad"})
        self.assertEqual(result, ":(")
    
    def test_default_mood(self):
        result = get_smiley.invoke({})
        self.assertEqual(result, ":)")
    
    def test_invalid_mood(self):
        with self.assertRaises(ValueError):
            get_smiley.invoke({"mood": "angry"})

if __name__ == '__main__':
    unittest.main()
```
"""

# Get all available tools
tools = utils.all_tool_functions()


def get_memory_context(task_description: str) -> tuple[str, str]:
    """
    Retrieve relevant memories and templates for the task.

    Args:
        task_description (str): Description of the tool to create.

    Returns:
        tuple[str, str]: (memory_context, template_context)
    """
    memory_context = ""
    template_context = ""

    if not MEMORY_AVAILABLE or not memory_manager:
        return memory_context, template_context

    try:
        # Try different memory retrieval methods
        relevant_memories = []

        # Use the MemoryManager instance methods
        if hasattr(memory_manager_instance, "retrieve_memories"):
            relevant_memories = memory_manager_instance.retrieve_memories(limit=5)
        elif hasattr(memory_manager_instance, "search_memories"):
            relevant_memories = memory_manager_instance.search_memories(
                query=task_description, limit=5
            )

        if relevant_memories:
            memory_context = "Relevant context from past interactions:\n"
            memory_context += "\n".join([f"- {mem}" for mem in relevant_memories])
            memory_context += "\n\n---\n\n"

        # Try to get tool templates
        tool_templates = []
        if hasattr(memory_manager_instance, "retrieve_memories"):
            tool_templates = memory_manager_instance.retrieve_memories(memory_type="tool_template", limit=1)

        if tool_templates:
            template_context = f"Relevant template found (use as starting point):\n```python\n{tool_templates[0]}\n```\n\n---\n\n"

    except Exception as e:
        logger.warning(f"Failed to retrieve memories: {e}")

    return memory_context, template_context


def check_shortcut_solution(task_description: str, memories: list) -> Optional[str]:
    """
    Check if there's a shortcut solution in the memories.

    Args:
        task_description (str): The task description.
        memories (list): List of relevant memories.
    """
    try:
        formatted_memories = "\n".join([f"- {mem}" for mem in memories])
        shortcut_prompt = f"""Analyze the following tool creation task and retrieved memories.

Task: {task_description}

Memories:
{formatted_memories}

Based ONLY on the provided memories, is there a direct pre-existing solution (e.g., identical tool already created) that fully satisfies this request?

Respond ONLY with:
- "SHORTCUT: [Explanation]" if a clear shortcut exists
- "NO SHORTCUT" if no clear shortcut is found"""

        response = config.default_langchain_model.invoke(
            [SystemMessage(content=shortcut_prompt)]
        )
        shortcut_content = response.content
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

        if isinstance(shortcut_text, str) and shortcut_text.startswith("SHORTCUT:"):
            return shortcut_text.replace("SHORTCUT:", "").strip()

    except Exception as e:
        logger.warning(f"Failed to check for shortcuts: {e}")

    return None


def reasoning(state: MessagesState) -> dict:
    """
    Tool Maker's reasoning step. Invokes the LLM with its prompt and current state.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        dict: Update for the graph state with the LLM's response (AIMessage).
    """
    print("\ntool_maker is thinking...")
    current_messages = state["messages"]

    # Extract current task/query
    task_description = "Create a tool based on request"
    query_context = "Initial tool creation request"

    if current_messages:
        for msg in reversed(current_messages):
            if isinstance(msg, HumanMessage):
                query_context = msg.content
                task_description = msg.content
                break

    # Ensure task_description is a string
    if isinstance(task_description, list) and len(task_description) > 0:
        if isinstance(task_description[0], str):
            task_description = task_description[0]
        elif isinstance(task_description[0], dict) and task_description[0].get("type") == "text":
            task_description = str(task_description[0].get("text", ""))
        else:
            task_description = str(task_description[0])
    elif not isinstance(task_description, str):
        task_description = str(task_description)

    print(f"Task: {task_description[:100]}...")

    # Get memory context
    memory_context, template_context = get_memory_context(task_description)

    # Check for shortcuts
    if memory_context:
        memories_list = memory_context.split("- ")[1:] if "- " in memory_context else []
        shortcut = check_shortcut_solution(task_description, memories_list)
        if shortcut:
            print(f"Shortcut found: {shortcut}")
            return {"messages": [AIMessage(content=shortcut, tool_calls=[])]}

    # Prepare messages for LLM
    messages_for_llm = list(current_messages)

    # Add context to system message
    combined_context = template_context + memory_context
    if combined_context and messages_for_llm:
        if isinstance(messages_for_llm[0], SystemMessage):
            original_content = messages_for_llm[0].content
            if not any(
                marker in original_content
                for marker in ["Relevant template found", "Relevant context from past"]
            ):
                messages_for_llm[0] = SystemMessage(
                    content=f"{combined_context}{original_content}"
                )
        else:
            messages_for_llm.insert(0, SystemMessage(content=combined_context.strip()))

    # Main LLM call with retry logic
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            if config.default_langchain_model is None:
                raise ValueError("Default language model not initialized")

            if not messages_for_llm:
                raise ValueError("No messages to process")

            # Use Google Gemini for tool calling from hybrid configuration
            tool_model = config.get_model_for_tools()
            if tool_model is None:
                # Fallback to default model if hybrid setup fails
                tool_model = config.default_langchain_model
                logger.warning("Using fallback model for tools - may not support function calling")
            
            tooled_up_model = tool_model.bind_tools(tools)
            response = tooled_up_model.invoke(messages_for_llm)
            return {"messages": [response]}

        except (HTTPError, RequestException) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"OpenRouter API Error. Retry {attempt + 1}/{max_retries}"
                )
                print(f"API error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error("Max retries reached for API Error")
                raise e

        except Exception as e:
            logger.error(
                f"Unexpected error in Tool Maker reasoning: {e}", exc_info=True
            )
            raise e

    # Fallback return if all retries fail
    return {"messages": [AIMessage(content="Tool Maker failed to generate a response.", tool_calls=[])]}


def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    """
    Checks the last message for tool calls and routes accordingly.

    Args:
        state (MessagesState): The current graph state.

    Returns:
        Literal["tools", END]: The next node to transition to.
    """
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    # Only access .tool_calls on AIMessage
    from langchain_core.messages import AIMessage
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Print thoughts if available
        if hasattr(last_message, "content") and last_message.content:
            content = last_message.content
            if isinstance(content, str) and content.strip():
                print("tool_maker thought:")
                print(content)
            elif isinstance(content, list) and content and isinstance(content[0], str):
                if content[0].strip():
                    print("tool_maker thought:")
                    print(content[0])

        print(
            f"\ntool_maker is using tools: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}"
        )
        return "tools"

    return END


def analyze_tool_creation_result(
    messages_history: list,
) -> tuple[str, Optional[str], str]:
    """
    Analyze the message history to determine if tool creation was successful.

    Args:
        messages_history (list): List of messages from the workflow.

    Returns:
        tuple[str, Optional[str], str]: (status, tool_name, final_message)
    """
    status = "failure"
    tool_name = None
    final_message = "Tool creation failed - unknown error"

    if not messages_history:
        return status, tool_name, "No messages in history"

    # Get the final message
    last_message = messages_history[-1]
    if hasattr(last_message, "content"):
        final_message = str(last_message.content)

    # Look for success indicators in the final message
    success_patterns = [
        r"successfully created.*?tool.*?['\"]([^'\"]+)['\"]",
        r"tool.*?['\"]([^'\"]+)['\"].*?successfully",
        r"created and tested.*?['\"]([^'\"]+)['\"]",
    ]

    for pattern in success_patterns:
        match = re.search(pattern, final_message.lower())
        if match:
            potential_tool_name = match.group(1)
            if potential_tool_name:
                status = "success"
                tool_name = potential_tool_name
                break

    # Check for explicit failure indicators
    failure_indicators = ["failed", "error", "exception", "cannot", "unable"]
    if any(indicator in final_message.lower() for indicator in failure_indicators):
        status = "failure"
        tool_name = None

    # Analyze command outputs for failures
    critical_failure = False
    for i, msg in enumerate(messages_history):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == "run_shell_command":
                    command = tool_call.get("args", {}).get("command", "")

                    # Check next message for tool output
                    if i + 1 < len(messages_history):
                        next_msg = messages_history[i + 1]
                        if isinstance(next_msg, ToolMessage):
                            output = str(next_msg.content).lower()

                            # Check for linting failures
                            if "ruff check" in command and (
                                "error:" in output or "failed" in output
                            ):
                                if "fixed" not in output:
                                    critical_failure = True
                                    break

                            # Check for test failures
                            if "unittest" in command:
                                if any(
                                    fail_indicator in output
                                    for fail_indicator in [
                                        "fail:",
                                        "error:",
                                        "failures=",
                                        "errors=",
                                    ]
                                ):
                                    if not (
                                        "failures=0" in output and "errors=0" in output
                                    ):
                                        critical_failure = True
                                        break

            if critical_failure:
                break

    if critical_failure:
        status = "failure"
        tool_name = None
        final_message = (
            f"Tool creation failed due to linting or testing errors. {final_message}"
        )

    return status, tool_name, final_message


# Create workflow
acting = ToolNode(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges(
    "reasoning", 
    check_for_tool_calls,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "reasoning")
graph = workflow.compile()


def tool_maker(task: str) -> Dict[str, Any]:
    """
    Executes the Tool Maker workflow to create a new tool.

    Args:
        task (str): The description of the tool to be created.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'status' (str): 'success' or 'failure'
            - 'result' (str | None): The name of the created tool on success
            - 'message' (str): The final message from the workflow
    """
    logger.info(f"Tool Maker invoked for task: {task[:100]}...")

    try:
        # Execute the workflow
        final_state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task),
                ]
            }
        )

        logger.info("Tool Maker graph invocation completed")

        # Extract messages history
        messages_history = final_state.get("messages", []) if final_state else []

        # Analyze the result
        status, tool_name, final_message = analyze_tool_creation_result(
            messages_history
        )

        # Update manifest on success
        if status == "success" and tool_name:
            module_path = os.path.join("tools", f"{tool_name}.py")
            manifest_entry = {
                "name": tool_name,
                "module_path": module_path,
                "function_name": tool_name,
                "description": f"Tool '{tool_name}' created for: {task[:100]}...",
            }

            if utils.add_manifest_entry(manifest_type="tool", entry=manifest_entry):
                logger.info(f"Successfully registered tool: {tool_name}")
                final_message = (
                    f"Successfully created, tested, and registered tool: '{tool_name}'"
                )
            else:
                logger.error(f"Failed to register tool: {tool_name}")
                status = "failure"
                tool_name = None
                final_message = f"Tool created but failed to register in manifest"

        result = {"status": status, "result": tool_name, "message": final_message}

        print(f"Tool Maker finished - Status: {status}, Tool: {tool_name}")
        logger.info(f"Tool Maker result: {result}")

        return result

    except Exception as e:
        error_msg = f"Tool Maker execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return {"status": "failure", "result": None, "message": error_msg}
