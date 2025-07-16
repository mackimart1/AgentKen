"""
Legacy Integration Module for Inferra V Enhanced System
Bridges existing agents and tools with the new enhanced architecture.
"""

import os
import sys
import json
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import utils
    import config
    import memory_manager
except ImportError as e:
    logging.warning(f"Could not import legacy modules: {e}")
    utils = None
    config = None
    memory_manager = None

from .tool_integration_system import BaseTool, ToolDefinition, ToolRegistry
from .agent_framework import BaseAgent, AgentCapability, Message, MessageType


@dataclass
class LegacyAgentInfo:
    """Information about a legacy agent"""

    name: str
    module_path: str
    function_name: str
    description: str
    capabilities: List[str]
    file_path: str


@dataclass
class LegacyToolInfo:
    """Information about a legacy tool"""

    name: str
    module_path: str
    function_name: str
    description: str
    parameters: Dict[str, Any]
    file_path: str


class LegacyToolWrapper(BaseTool):
    """Wrapper to integrate legacy tools with the enhanced system"""

    def __init__(self, legacy_tool_info: LegacyToolInfo, legacy_function: Callable):
        self.legacy_info = legacy_tool_info
        self.legacy_function = legacy_function

        # Create tool definition from legacy tool
        definition = ToolDefinition(
            name=legacy_tool_info.name,
            version="1.0.0",
            description=legacy_tool_info.description,
            input_schema=self._extract_input_schema(),
            output_schema={"result": "any", "status": "string", "message": "string"},
            timeout=60.0,
            max_retries=3,
        )

        super().__init__(definition)

    def _extract_input_schema(self) -> Dict[str, Any]:
        """Extract input schema from legacy function signature"""
        try:
            sig = inspect.signature(self.legacy_function)
            schema = {}

            for param_name, param in sig.parameters.items():
                if param_name in ["self", "cls"]:
                    continue

                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"

                schema[param_name] = param_type

            return schema
        except Exception as e:
            logging.warning(
                f"Could not extract schema for {self.legacy_info.name}: {e}"
            )
            return {"input": "string"}

    def _execute(self, **kwargs) -> Any:
        """Execute the legacy tool function"""
        try:
            # Call the legacy function
            result = self.legacy_function(**kwargs)

            # Normalize the result format
            if isinstance(result, dict) and "status" in result:
                return result
            else:
                return {
                    "status": "success",
                    "result": result,
                    "message": f"Tool {self.legacy_info.name} executed successfully",
                }
        except Exception as e:
            return {
                "status": "failure",
                "result": None,
                "message": f"Tool {self.legacy_info.name} failed: {str(e)}",
            }


class LegacyAgentWrapper(BaseAgent):
    """Wrapper to integrate legacy agents with the enhanced system"""

    def __init__(
        self, legacy_agent_info: LegacyAgentInfo, legacy_function: Callable, message_bus
    ):
        self.legacy_info = legacy_agent_info
        self.legacy_function = legacy_function

        # Create capabilities from legacy agent info
        capabilities = []
        for cap_name in legacy_agent_info.capabilities:
            capability = AgentCapability(
                name=cap_name,
                description=f"Legacy capability: {cap_name}",
                input_schema={"task": "string"},
                output_schema={
                    "status": "string",
                    "result": "any",
                    "message": "string",
                },
            )
            capabilities.append(capability)

        super().__init__(legacy_agent_info.name, capabilities, message_bus)

    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute legacy agent capability"""
        try:
            # Extract task from payload
            task = payload.get("task", payload.get("input", ""))

            # Call the legacy agent function
            result = self.legacy_function(task)

            # Normalize the result format
            if isinstance(result, dict) and "status" in result:
                return result
            else:
                return {
                    "status": "success",
                    "result": result,
                    "message": f"Agent {self.legacy_info.name} completed task successfully",
                }
        except Exception as e:
            return {
                "status": "failure",
                "result": None,
                "message": f"Agent {self.legacy_info.name} failed: {str(e)}",
            }


class LegacyIntegrationManager:
    """Manages integration of legacy agents and tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.legacy_agents: Dict[str, LegacyAgentInfo] = {}
        self.legacy_tools: Dict[str, LegacyToolInfo] = {}
        self.wrapped_agents: Dict[str, LegacyAgentWrapper] = {}
        self.wrapped_tools: Dict[str, LegacyToolWrapper] = {}

        # Paths to legacy components
        self.agents_path = project_root / "agents"
        self.tools_path = project_root / "tools"

    def discover_legacy_components(self):
        """Discover all legacy agents and tools"""
        self.logger.info("Discovering legacy components...")

        # Discover agents
        self._discover_legacy_agents()

        # Discover tools
        self._discover_legacy_tools()

        self.logger.info(
            f"Discovered {len(self.legacy_agents)} agents and {len(self.legacy_tools)} tools"
        )

    def _discover_legacy_agents(self):
        """Discover legacy agents from the agents directory"""
        if not self.agents_path.exists():
            self.logger.warning(f"Agents directory not found: {self.agents_path}")
            return

        # Try to load agents manifest
        manifest_path = project_root / "agents_manifest.json"
        agents_from_manifest = {}

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(manifest, list):
                        agents_from_manifest = {
                            agent["name"]: agent for agent in manifest
                        }
                    elif isinstance(manifest, dict):
                        agents_from_manifest = manifest.get("agents", {})
                    else:
                        agents_from_manifest = {}
                self.logger.info(
                    f"Loaded agents manifest with {len(agents_from_manifest)} entries"
                )
            except Exception as e:
                self.logger.warning(f"Could not load agents manifest: {e}")

        # Scan agents directory
        for agent_file in self.agents_path.glob("*.py"):
            if agent_file.name.startswith("__"):
                continue

            agent_name = agent_file.stem

            # Get info from manifest or infer
            if agent_name in agents_from_manifest:
                agent_info = agents_from_manifest[agent_name]
                description = agent_info.get(
                    "description", f"Legacy agent: {agent_name}"
                )
                capabilities = agent_info.get("capabilities", [agent_name])
            else:
                description = f"Legacy agent: {agent_name}"
                capabilities = [agent_name]

            legacy_agent = LegacyAgentInfo(
                name=agent_name,
                module_path=f"agents.{agent_name}",
                function_name=agent_name,
                description=description,
                capabilities=capabilities,
                file_path=str(agent_file),
            )

            self.legacy_agents[agent_name] = legacy_agent
            self.logger.debug(f"Discovered agent: {agent_name}")

    def _discover_legacy_tools(self):
        """Discover legacy tools from the tools directory"""
        if not self.tools_path.exists():
            self.logger.warning(f"Tools directory not found: {self.tools_path}")
            return

        # Try to load tools manifest
        manifest_path = project_root / "tools_manifest.json"
        tools_from_manifest = {}

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(manifest, list):
                        tools_from_manifest = {tool["name"]: tool for tool in manifest}
                    elif isinstance(manifest, dict):
                        tools_from_manifest = manifest.get("tools", {})
                    else:
                        tools_from_manifest = {}
                self.logger.info(
                    f"Loaded tools manifest with {len(tools_from_manifest)} entries"
                )
            except Exception as e:
                self.logger.warning(f"Could not load tools manifest: {e}")

        # Scan tools directory
        for tool_file in self.tools_path.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue

            tool_name = tool_file.stem

            # Get info from manifest or infer
            if tool_name in tools_from_manifest:
                tool_info = tools_from_manifest[tool_name]
                description = tool_info.get("description", f"Legacy tool: {tool_name}")
                parameters = tool_info.get("parameters", {})
            else:
                description = f"Legacy tool: {tool_name}"
                parameters = {}

            legacy_tool = LegacyToolInfo(
                name=tool_name,
                module_path=f"tools.{tool_name}",
                function_name=tool_name,
                description=description,
                parameters=parameters,
                file_path=str(tool_file),
            )

            self.legacy_tools[tool_name] = legacy_tool
            self.logger.debug(f"Discovered tool: {tool_name}")

    def load_legacy_function(
        self, module_path: str, function_name: str
    ) -> Optional[Callable]:
        """Load a legacy function from module path"""
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, function_name):
                return getattr(module, function_name)
            else:
                self.logger.warning(
                    f"Function {function_name} not found in {module_path}"
                )
                return None
        except Exception as e:
            self.logger.error(f"Could not load {module_path}.{function_name}: {e}")
            return None

    def wrap_legacy_agents(self, message_bus) -> Dict[str, LegacyAgentWrapper]:
        """Wrap legacy agents for the enhanced system"""
        self.logger.info("Wrapping legacy agents...")

        for agent_name, agent_info in self.legacy_agents.items():
            try:
                # Load the legacy function
                legacy_function = self.load_legacy_function(
                    agent_info.module_path, agent_info.function_name
                )

                if legacy_function:
                    # Create wrapper
                    wrapper = LegacyAgentWrapper(
                        agent_info, legacy_function, message_bus
                    )
                    self.wrapped_agents[agent_name] = wrapper
                    self.logger.info(f"Wrapped agent: {agent_name}")
                else:
                    self.logger.warning(
                        f"Could not load function for agent: {agent_name}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to wrap agent {agent_name}: {e}")

        return self.wrapped_agents

    def wrap_legacy_tools(self) -> Dict[str, LegacyToolWrapper]:
        """Wrap legacy tools for the enhanced system"""
        self.logger.info("Wrapping legacy tools...")

        for tool_name, tool_info in self.legacy_tools.items():
            try:
                # Load the legacy function
                legacy_function = self.load_legacy_function(
                    tool_info.module_path, tool_info.function_name
                )

                if legacy_function:
                    # Create wrapper
                    wrapper = LegacyToolWrapper(tool_info, legacy_function)
                    self.wrapped_tools[tool_name] = wrapper
                    self.logger.info(f"Wrapped tool: {tool_name}")
                else:
                    self.logger.warning(
                        f"Could not load function for tool: {tool_name}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to wrap tool {tool_name}: {e}")

        return self.wrapped_tools

    def register_with_enhanced_system(
        self, tool_registry: ToolRegistry, agent_list: List
    ):
        """Register wrapped components with the enhanced system"""
        self.logger.info("Registering legacy components with enhanced system...")

        # Register tools
        for tool_name, wrapped_tool in self.wrapped_tools.items():
            try:
                tool_registry.register_tool(wrapped_tool)
                self.logger.info(f"Registered tool: {tool_name}")
            except Exception as e:
                self.logger.error(f"Failed to register tool {tool_name}: {e}")

        # Add agents to agent list
        for agent_name, wrapped_agent in self.wrapped_agents.items():
            try:
                agent_list.append(wrapped_agent)
                self.logger.info(f"Registered agent: {agent_name}")
            except Exception as e:
                self.logger.error(f"Failed to register agent {agent_name}: {e}")

    def get_integration_report(self) -> Dict[str, Any]:
        """Get a report on the integration status"""
        return {
            "discovered_agents": len(self.legacy_agents),
            "discovered_tools": len(self.legacy_tools),
            "wrapped_agents": len(self.wrapped_agents),
            "wrapped_tools": len(self.wrapped_tools),
            "agent_names": list(self.legacy_agents.keys()),
            "tool_names": list(self.legacy_tools.keys()),
            "wrapped_agent_names": list(self.wrapped_agents.keys()),
            "wrapped_tool_names": list(self.wrapped_tools.keys()),
        }


def integrate_legacy_components(
    tool_registry: ToolRegistry, agent_list: List, message_bus
) -> Dict[str, Any]:
    """
    Main function to integrate all legacy components

    Args:
        tool_registry: Enhanced tool registry
        agent_list: List to add wrapped agents to
        message_bus: Message bus for agent communication

    Returns:
        Integration report
    """
    manager = LegacyIntegrationManager()

    # Discover legacy components
    manager.discover_legacy_components()

    # Wrap components
    manager.wrap_legacy_agents(message_bus)
    manager.wrap_legacy_tools()

    # Register with enhanced system
    manager.register_with_enhanced_system(tool_registry, agent_list)

    # Return integration report
    return manager.get_integration_report()


# Utility functions for backward compatibility
def get_legacy_agent_function(agent_name: str) -> Optional[Callable]:
    """Get a legacy agent function by name"""
    try:
        if utils and hasattr(utils, "get_agent_details"):
            agent_details = utils.get_agent_details(agent_name)
            if agent_details:
                return utils.load_registered_module(agent_details)
    except Exception as e:
        logging.warning(f"Could not load legacy agent {agent_name}: {e}")

    return None


def get_legacy_tool_function(tool_name: str) -> Optional[Callable]:
    """Get a legacy tool function by name"""
    try:
        if utils and hasattr(utils, "all_tool_functions"):
            tools = utils.all_tool_functions()
            for tool in tools:
                if hasattr(tool, "name") and tool.name == tool_name:
                    return tool
                elif hasattr(tool, "__name__") and tool.__name__ == tool_name:
                    return tool
    except Exception as e:
        logging.warning(f"Could not load legacy tool {tool_name}: {e}")

    return None


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)

    manager = LegacyIntegrationManager()
    manager.discover_legacy_components()

    print("Integration Report:")
    print(json.dumps(manager.get_integration_report(), indent=2))
