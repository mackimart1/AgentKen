"""
Agent Smith: The agent architect and developer for the AgentK system.

Responsible for creating new agents based on task requirements provided by Hermes.
It plans the agent design, requests necessary tools from ToolMaker, writes the
agent code and tests, formats/lints the code, runs the tests, and reports
the outcome back to Hermes. Uses LangGraph for its internal workflow.
"""

from typing import Literal, Optional, List, Dict, Any
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

import config
import utils
import memory_manager

# Setup logger for agent_smith
logger = logging.getLogger(__name__)

# Initialize memory manager
memory_manager_instance = memory_manager.MemoryManager()


class AgentCreationPhase(Enum):
    """Enum to track the current phase of agent creation."""

    PLANNING = "planning"
    TOOL_CREATION = "tool_creation"
    AGENT_WRITING = "agent_writing"
    FORMATTING = "formatting"
    LINTING = "linting"
    TESTING = "testing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class AgentCreationState:
    """Enhanced state tracking for agent creation process."""

    is_complete: bool = False
    agent_name: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    max_duration: int = 600  # Increased to 10 minutes
    max_inactivity: int = 120  # Increased to 2 minutes
    current_phase: AgentCreationPhase = AgentCreationPhase.PLANNING
    files_written: List[str] = field(default_factory=list)
    tools_requested: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    # Loop detection
    reasoning_count: int = 0
    max_reasoning_iterations: int = 50
    last_reasoning_content: str = ""
    repeated_content_count: int = 0
    max_repeated_content: int = 3

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()

    def advance_phase(self, new_phase: AgentCreationPhase):
        """Advance to the next phase and update activity."""
        self.current_phase = new_phase
        self.update_activity()
        logger.info(f"Agent creation phase advanced to: {new_phase.value}")

    def add_error(self, error: str):
        """Add an error to the error list."""
        self.errors.append(f"[{self.current_phase.value}] {error}")
        logger.error(f"Agent creation error in {self.current_phase.value}: {error}")

    def check_timeout(self) -> tuple[bool, Optional[str]]:
        """Check if the agent creation has timed out."""
        current_time = datetime.now()
        total_duration = (current_time - self.start_time).total_seconds()
        inactivity_duration = (current_time - self.last_activity).total_seconds()

        if total_duration > self.max_duration:
            return True, f"Agent creation timed out after {self.max_duration} seconds"
        if inactivity_duration > self.max_inactivity:
            return (
                True,
                f"Agent creation stalled after {self.max_inactivity} seconds of inactivity",
            )
        return False, None

    def should_retry(self) -> bool:
        """Check if we should retry the current operation."""
        return self.retry_count < self.max_retries

    def increment_retry(self):
        """Increment the retry counter."""
        self.retry_count += 1

    def reset_retry(self):
        """Reset the retry counter for a new operation."""
        self.retry_count = 0


# Initialize completion state
creation_state = AgentCreationState()

# Enhanced system prompt with better structure and clearer instructions
system_prompt = """You are agent_smith, a ReAct agent that develops other ReAct agents for the AgentK system.

CONTEXT:
AgentK is a self-evolving AGI made of agents that collaborate and build new agents as needed to complete tasks.
You are responsible for creating high-quality, well-tested agents that integrate seamlessly into the AgentK ecosystem.

YOUR WORKFLOW (CRITICAL - Follow in Order):

1. PLANNING PHASE
   - Analyze the task requirements thoroughly
   - Design the agent architecture and identify required capabilities
   - Determine what tools are needed (existing or new)

2. TOOL CREATION PHASE (if needed)
   - Use `assign_agent_to_task` to request new tools from tool_maker
   - Wait for confirmation that tools are created before proceeding

3. AGENT WRITING PHASE
   - Write the agent implementation to `agents/agent_name.py`
   - Write a comprehensive test to `tests/agents/test_agent_name.py`
   - **CRITICAL**: Verify files are written successfully by checking tool outputs

4. CODE QUALITY PHASE
   - Format code: `black agents/agent_name.py tests/agents/test_agent_name.py`
   - Lint code: `ruff check agents/agent_name.py tests/agents/test_agent_name.py --fix`
   - **CRITICAL**: Fix any linting errors that cannot be auto-fixed

5. TESTING PHASE
   - Run tests: `python -m unittest tests/agents/test_agent_name.py -v`
   - **CRITICAL**: Debug and fix any test failures
   - Ensure all tests pass before proceeding

6. COMPLETION PHASE
   - Only after ALL previous steps succeed, generate final success message
   - Format: "Successfully created, formatted, linted, and tested agent: 'agent_name'"

CRITICAL REQUIREMENTS:
- **ALWAYS** examine tool outputs carefully for success/failure indicators
- **NEVER** proceed to next phase if current phase has errors
- **ALWAYS** retry failed operations up to 3 times before giving up
- **NEVER** generate success message unless ALL steps verified successful
- Use proper error handling and provide detailed error messages

AGENT DESIGN PRINCIPLES:
- Agents should be general-purpose, not overly specific function wrappers
- Follow existing patterns in the `agents/` directory
- Import tools from `tools/` directory: `from tools.tool_name import tool_name`
- Use LangGraph for agent workflow definition
- Include comprehensive error handling and logging

CRITICAL AGENT FORMAT REQUIREMENTS:
- Agent function MUST accept a 'task' parameter: `def agent_name(task: str) -> Dict[str, Any]:`
- Agent function MUST return a dictionary with these keys:
  - 'status': 'success' or 'failure'
  - 'result': The actual result/output of the agent
  - 'message': A descriptive message about what was done
- Example agent structure:
```python
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def agent_name(task: str) -> Dict[str, Any]:
    \"\"\"
    Agent description here.
    
    Args:
        task (str): The task to perform
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
    \"\"\"
    try:
        # Agent logic here
        result = "agent output"
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully completed task: {task}"
        }
    except Exception as e:
        logger.error(f"Error in {agent_name.__name__}: {e}")
        return {
            "status": "failure", 
            "result": None,
            "message": f"Failed to complete task: {str(e)}"
        }
```

If you encounter errors, analyze them carefully and fix the underlying issues.
If you need to retry an operation, clearly state what you're retrying and why.
"""


class AgentSmithWorkflow:
    """Enhanced workflow management for Agent Smith."""

    def __init__(self, tools: List[Any]):
        self.tools = tools
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow with increased recursion limit."""
        workflow = StateGraph(MessagesState)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("reasoning")
        workflow.add_conditional_edges(
            "reasoning", 
            self._check_for_tool_calls,
            {
                "tools": "tools",
                "END": END
            }
        )
        workflow.add_edge("tools", "reasoning")
        # Compile workflow (recursion limit will be set during invoke)
        return workflow.compile()

    def _reasoning_node(self, state: MessagesState) -> Dict[str, Any]:
        """Enhanced reasoning step with better error handling and loop detection."""
        # Increment reasoning counter for loop detection
        creation_state.reasoning_count += 1
        
        logger.info(
            f"Agent Smith reasoning #{creation_state.reasoning_count} in phase: {creation_state.current_phase.value}"
        )

        # Check for too many reasoning iterations (infinite loop detection)
        if creation_state.reasoning_count > creation_state.max_reasoning_iterations:
            error_msg = f"Maximum reasoning iterations ({creation_state.max_reasoning_iterations}) exceeded. Possible infinite loop detected."
            creation_state.add_error(error_msg)
            creation_state.advance_phase(AgentCreationPhase.FAILED)
            return {"messages": [AIMessage(content=f"Error: {error_msg}")]}

        # Check for timeout conditions
        is_timeout, timeout_message = creation_state.check_timeout()
        if is_timeout:
            creation_state.advance_phase(AgentCreationPhase.FAILED)
            return {"messages": [AIMessage(content=f"Error: {timeout_message}")]}

        # Update activity timestamp
        creation_state.update_activity()

        # Add context about current phase to the conversation
        phase_context = self._get_phase_context()
        messages = state["messages"] + [SystemMessage(content=phase_context)]

        try:
            # Use Google Gemini for tool calling from hybrid configuration
            tool_model = config.get_model_for_tools()
            if tool_model is None:
                # Fallback to default model if hybrid setup fails
                tool_model = config.default_langchain_model
                logger.warning("Using fallback model for tools - may not support function calling")
            
            tooled_up_model = tool_model.bind_tools(self.tools)
            response = tooled_up_model.invoke(messages)

            # Process the response only if it is an AIMessage
            if isinstance(response, AIMessage):
                # Check for repeated content (another loop detection mechanism)
                content = self._extract_content(response)
                if content == creation_state.last_reasoning_content:
                    creation_state.repeated_content_count += 1
                    logger.warning(f"Repeated content detected ({creation_state.repeated_content_count}/{creation_state.max_repeated_content})")
                    
                    if creation_state.repeated_content_count >= creation_state.max_repeated_content:
                        error_msg = "Agent is repeating the same response. Stopping to prevent infinite loop."
                        creation_state.add_error(error_msg)
                        creation_state.advance_phase(AgentCreationPhase.FAILED)
                        return {"messages": [AIMessage(content=f"Error: {error_msg}")]}
                else:
                    creation_state.repeated_content_count = 0
                    creation_state.last_reasoning_content = content
                
                self._process_response(response)

            return {"messages": [response]}

        except Exception as e:
            error_msg = f"Error in reasoning step: {str(e)}"
            creation_state.add_error(error_msg)
            logger.error(error_msg, exc_info=True)
            return {"messages": [AIMessage(content=f"Internal error: {error_msg}")]}

    def _get_phase_context(self) -> str:
        """Get context about the current phase for the LLM."""
        context = f"""
CURRENT PHASE: {creation_state.current_phase.value}
REASONING ITERATION: {creation_state.reasoning_count}/{creation_state.max_reasoning_iterations}
FILES WRITTEN: {', '.join(creation_state.files_written) if creation_state.files_written else 'None'}
TOOLS REQUESTED: {', '.join(creation_state.tools_requested) if creation_state.tools_requested else 'None'}
RETRY COUNT: {creation_state.retry_count}/{creation_state.max_retries}
"""

        if creation_state.errors:
            context += f"\nRECENT ERRORS: {'; '.join(creation_state.errors[-3:])}"
        
        # Add warning if approaching limits
        if creation_state.reasoning_count > creation_state.max_reasoning_iterations * 0.8:
            context += f"\nWARNING: Approaching maximum reasoning iterations. Please ensure progress is being made."
        
        if creation_state.repeated_content_count > 0:
            context += f"\nWARNING: Repeated content detected {creation_state.repeated_content_count} times. Ensure you're making progress."

        return context

    def _process_response(self, response: AIMessage):
        """Process the AI response to update state accordingly."""
        if not isinstance(response, AIMessage):
            return

        content = self._extract_content(response)

        # Track file creation from tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.get("name") == "write_to_file":
                    file_path = tool_call.get("args", {}).get("file", "")
                    if file_path and file_path not in creation_state.files_written:
                        creation_state.files_written.append(file_path)
                        logger.info(f"Tracked file creation: {file_path}")

        # Check for completion message
        if (
            "successfully created, formatted, linted, and tested agent"
            in content.lower()
        ):
            creation_state.advance_phase(AgentCreationPhase.COMPLETE)
            creation_state.is_complete = True

            # Extract agent name
            try:
                if "agent: '" in content:
                    name_part = content.split("agent: '")[1]
                    creation_state.agent_name = name_part.split("'")[0]
            except (IndexError, AttributeError):
                logger.warning("Could not extract agent name from completion message")

    def _extract_content(self, response: AIMessage) -> str:
        """Extract string content from AI message."""
        content = response.content
        if isinstance(content, list):
            content = next(
                (item for item in content if isinstance(item, str)), str(content)
            )
        return content if isinstance(content, str) else ""

    def _check_for_tool_calls(self, state: MessagesState) -> Literal["tools", "END"]:
        """Enhanced tool call checking with state management."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Check for completion state
        if (
            creation_state.is_complete
            or creation_state.current_phase == AgentCreationPhase.COMPLETE
        ):
            logger.info(
                f"Agent creation completed successfully: {creation_state.agent_name}"
            )
            return "END"

        # Check for failure state
        if creation_state.current_phase == AgentCreationPhase.FAILED:
            logger.error("Agent creation failed")
            return "END"

        # Check for timeout or error messages
        if isinstance(last_message, AIMessage):
            content = self._extract_content(last_message)
            if "Error:" in content or "timeout" in content.lower():
                creation_state.advance_phase(AgentCreationPhase.FAILED)
                return "END"

        # Check for tool calls
        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "tools"

        return "END"


def _ensure_directories():
    """Ensure required directories exist."""
    directories = ["agents", "tools", "tests/agents", "tests/tools"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created missing directory: {directory}")


def _load_and_validate_tools() -> List[Any]:
    """Load and validate all required tools."""
    # Dynamically load all tools
    tools = utils.all_tool_functions()

    # Get loaded tool names
    loaded_tool_names = set()
    for tool in tools:
        if hasattr(tool, "name"):
            loaded_tool_names.add(tool.name)
        else:
            logger.warning(f"Tool object {type(tool)} missing '.name' attribute")

    # Required tools for Agent Smith
    required_tools = {
        "write_to_file",
        "read_file",
        "delete_file",
        "run_shell_command",
        "assign_agent_to_task",
        "list_available_agents",
    }

    # Check for missing tools and attempt manual loading
    missing_tools = required_tools - loaded_tool_names
    if missing_tools:
        logger.warning(f"Missing required tools: {missing_tools}")

        # Manual tool imports
        tool_imports = {
            "write_to_file": "tools.write_to_file",
            "read_file": "tools.read_file",
            "delete_file": "tools.delete_file",
            "run_shell_command": "tools.run_shell_command",
            "assign_agent_to_task": "tools.assign_agent_to_task",
            "list_available_agents": "tools.list_available_agents",
        }

        tools_dict = {
            getattr(t, "name", str(t)): t for t in tools if hasattr(t, "name")
        }

        for tool_name in missing_tools:
            if tool_name in tool_imports:
                try:
                    module_path = tool_imports[tool_name]
                    module = __import__(module_path, fromlist=[tool_name])
                    tool_func = getattr(module, tool_name)
                    tools_dict[tool_name] = tool_func
                    logger.info(f"Manually loaded tool: {tool_name}")
                except ImportError as e:
                    logger.error(f"Failed to load tool {tool_name}: {e}")

        tools = list(tools_dict.values())

    logger.info(f"Loaded {len(tools)} tools for Agent Smith")
    return tools


# Initialize the workflow
_ensure_directories()
tools = _load_and_validate_tools()
workflow_manager = AgentSmithWorkflow(tools)


def agent_smith(task: str) -> Dict[str, Any]:
    """
    Executes the Agent Smith workflow to design and implement a new agent.

    Args:
        task (str): The description of the agent to be created.

    Returns:
        Dict[str, Any]: A dictionary containing:
                       - 'status' (str): 'success' or 'failure'
                       - 'result' (str | None): The name of the created agent on success
                       - 'message' (str): The final message from the agent
                       - 'phase' (str): The final phase reached
                       - 'files_created' (List[str]): List of files created
                       - 'errors' (List[str]): List of errors encountered
    """
    # Reset state for new task
    global creation_state
    creation_state = AgentCreationState()

    logger.info(f"Starting agent creation task: {task}")

    try:
        # Execute the workflow with increased recursion limit
        final_state = workflow_manager.workflow.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task),
                ]
            },
            config={"recursion_limit": 100}
        )

        # Extract final message
        last_message_content = "No response generated"
        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            last_message_content = workflow_manager._extract_content(last_message)

        # Determine final status
        if (
            creation_state.is_complete
            and creation_state.current_phase == AgentCreationPhase.COMPLETE
        ):
            status = "success"
            result_data = creation_state.agent_name
        else:
            status = "failure"
            result_data = None
            if not creation_state.errors:
                creation_state.add_error("Agent creation incomplete - unknown reason")

        # Store memory of the creation attempt
        try:
            import json

            memory_data = {
                "task": task,
                "status": status,
                "phase": creation_state.current_phase.value,
                "files": creation_state.files_written,
                "errors": creation_state.errors,
            }

            # Convert to JSON string to ensure serialization compatibility
            memory_json = json.dumps(memory_data)

            memory_manager_instance.add_memory(
                key=f"agent_creation_{datetime.now().timestamp()}",
                value=memory_json,
                memory_type="agent_creation_log",
                agent_name="agent_smith",
            )
        except Exception as memory_error:
            logger.warning(f"Failed to store memory: {memory_error}")

        return {
            "status": status,
            "result": result_data,
            "message": last_message_content,
            "phase": creation_state.current_phase.value,
            "files_created": creation_state.files_written.copy(),
            "errors": creation_state.errors.copy(),
        }

    except Exception as e:
        error_msg = f"Critical error in agent_smith: {str(e)}"
        logger.error(error_msg, exc_info=True)
        creation_state.add_error(error_msg)

        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "phase": creation_state.current_phase.value,
            "files_created": creation_state.files_written.copy(),
            "errors": creation_state.errors.copy(),
        }
    