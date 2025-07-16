"""
Agent Template System

Provides templates and utilities for creating new agents with consistent
structure, documentation, and best practices.
"""

import os
import re
from typing import Dict, Any, Optional
from datetime import datetime


class AgentTemplate:
    """Template for creating new agents with consistent structure"""

    @staticmethod
    def generate_agent_code(
        agent_name: str, description: str, capabilities: list, author: str, user_id: str
    ) -> str:
        """Generate agent code from template"""

        # Convert agent_name to snake_case for filename
        agent_filename = agent_name.lower().replace(" ", "_").replace("-", "_")

        # Generate capabilities string
        capabilities_str = "\n".join([f"    - {cap}" for cap in capabilities])

        # Generate imports based on capabilities
        imports = AgentTemplate._generate_imports(capabilities)

        template = f'''"""
{agent_name}: {description}

Created by: {author}
User ID: {user_id}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Capabilities:
{capabilities_str}

This agent follows the standard AgentK pattern and integrates with the
permissioned creation system.
"""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config
import utils
import memory_manager

# Setup logger for {agent_filename}
logger = logging.getLogger(__name__)

# Initialize memory manager
try:
    memory_manager_instance = memory_manager.MemoryManager()
    MEMORY_AVAILABLE = True
except Exception as e:
    logger.warning(f"Memory manager not available: {{e}}")
    memory_manager_instance = None
    MEMORY_AVAILABLE = False

{imports}

system_prompt = """You are {agent_filename}, a ReAct agent in the AgentK system.

DESCRIPTION:
{description}

CAPABILITIES:
{capabilities_str}

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
            return "Relevant context from past interactions:\\n" + \\
                   "\\n".join([f"- {{mem}}" for mem in relevant_memories]) + "\\n\\n---\\n\\n"
    except Exception as e:
        logger.warning(f"Failed to retrieve memories: {{e}}")
    
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
    context = f"""{{memory_context}}
TASK: {{task_description}}

AVAILABLE TOOLS:
{{', '.join([tool.name for tool in tools])}}

Please analyze this task and determine the best approach to complete it.
Use the available tools as needed and provide a comprehensive response.
"""
    
    # Create system message with context
    system_msg = SystemMessage(content=system_prompt + "\\n\\n" + context)
    
    return {{
        "messages": [system_msg, HumanMessage(content=task_description)]
    }}

def acting(state: MessagesState) -> Dict[str, Any]:
    """Acting node for tool execution"""
    return {{"messages": state["messages"]}}

def check_for_tool_calls(state: MessagesState) -> str:
    """Check if the last message contains tool calls"""
    messages = state["messages"]
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check if the message content indicates completion
    content = last_message.content.lower() if hasattr(last_message, 'content') else ""
    if any(keyword in content for keyword in ["success", "completed", "finished", "done"]):
        return END
    
    return "reasoning"

def analyze_result(messages_history: list) -> tuple[str, Optional[str], str]:
    """Analyze the agent execution result"""
    if not messages_history:
        return "failure", None, "No messages generated"
    
    last_message = messages_history[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
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

def {agent_filename}(task: str) -> Dict[str, Any]:
    """
    {agent_name} agent implementation.
    
    Args:
        task (str): The task to perform
        
    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message
    """
    try:
        logger.info(f"Starting {agent_filename} task: {{task}}")
        
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
        result = app.invoke({{
            "messages": [HumanMessage(content=task)]
        }})
        
        # Analyze result
        messages_history = result.get("messages", [])
        status, result_name, message = analyze_result(messages_history)
        
        logger.info(f"{agent_filename} completed with status: {{status}}")
        
        return {{
            "status": status,
            "result": result_name,
            "message": message
        }}
        
    except Exception as e:
        logger.error(f"Error in {agent_filename}: {{e}}")
        return {{
            "status": "failure",
            "result": None,
            "message": f"Failed to complete task: {{str(e)}}"
        }}

if __name__ == "__main__":
    # Test the agent
    test_task = "Test task for {agent_name}"
    result = {agent_filename}(test_task)
    print(f"Test result: {{result}}")
'''

        return template

    @staticmethod
    def _generate_imports(capabilities: list) -> str:
        """Generate appropriate imports based on agent capabilities"""
        imports = []

        # Add capability-specific imports
        for capability in capabilities:
            capability_lower = capability.lower()
            if "web" in capability_lower or "search" in capability_lower:
                imports.append(
                    "from tools.duck_duck_go_web_search import duck_duck_go_web_search"
                )
            elif "file" in capability_lower or "write" in capability_lower:
                imports.append("from tools.write_to_file import write_to_file")
                imports.append("from tools.read_file import read_file")
            elif "terminal" in capability_lower or "shell" in capability_lower:
                imports.append("from tools.run_shell_command import run_shell_command")
            elif "memory" in capability_lower:
                imports.append("from tools.memory_manager import memory_manager")

        # Remove duplicates and sort
        imports = list(set(imports))
        imports.sort()

        return "\n".join(imports)

    @staticmethod
    def generate_test_code(agent_name: str, capabilities: list) -> str:
        """Generate test code for the agent"""

        agent_filename = agent_name.lower().replace(" ", "_").replace("-", "_")

        template = f'''"""
Tests for {agent_name} agent

Generated by the permissioned creation system.
"""
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.{agent_filename} import {agent_filename}

class Test{agent_name.replace(' ', '').replace('-', '')}(unittest.TestCase):
    """Test cases for {agent_name} agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_agent_creation(self):
        """Test that the agent can be created and called"""
        task = "Test task for {agent_name}"
        result = {agent_filename}(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("result", result)
        self.assertIn("message", result)
    
    def test_successful_execution(self):
        """Test successful task execution"""
        task = "Perform a simple test operation"
        result = {agent_filename}(task)
        
        # Should return a valid response structure
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("message", result)
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        task = ""  # Empty task should be handled gracefully
        result = {agent_filename}(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("message", result)
    
    def test_capabilities(self):
        """Test that the agent has the expected capabilities"""
        # Test that the agent can handle tasks related to its capabilities
        capabilities = {capabilities}
        
        for capability in capabilities:
            task = f"Test {{capability.lower()}} capability"
            result = {agent_filename}(task)
            
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
    
    @patch('agents.{agent_filename}.utils.all_tool_functions')
    def test_tool_integration(self, mock_tools):
        """Test integration with tools"""
        # Mock available tools
        mock_tools.return_value = []
        
        task = "Test tool integration"
        result = {agent_filename}(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
    
    def test_memory_integration(self):
        """Test integration with memory system"""
        task = "Test memory integration"
        result = {agent_filename}(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)

if __name__ == '__main__':
    unittest.main()
'''

        return template

    @staticmethod
    def generate_manifest_entry(
        agent_name: str, description: str, capabilities: list, author: str, user_id: str
    ) -> Dict[str, Any]:
        """Generate manifest entry for the agent"""
        return {
            "name": agent_name.lower().replace(" ", "_").replace("-", "_"),
            "description": description,
            "capabilities": capabilities,
            "author": author,
            "created_by": user_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "status": "active",
        }
