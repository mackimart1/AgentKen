#!/usr/bin/env python3
"""
Centralized Entity Creation System

Provides a unified interface for creating agents and tools with:
- Permission enforcement
- Template application
- Automated testing
- Documentation updates
- Audit logging
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.roles import (
    role_manager,
    Permission,
    UserRole,
    requires_permission,
    log_audit_event,
    create_default_admin,
)
from templates.agent_template import AgentTemplate
from templates.tool_template import ToolTemplate

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EntityCreator:
    """Centralized entity creation system"""

    def __init__(self):
        self.agents_dir = "agents"
        self.tools_dir = "tools"
        self.tests_dir = "tests"
        self.agents_manifest = "agents_manifest.json"
        self.tools_manifest = "tools_manifest.json"

        # Ensure directories exist
        self._ensure_directories()

        # Initialize default admin if needed
        create_default_admin()

    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.agents_dir,
            self.tools_dir,
            self.tests_dir,
            f"{self.tests_dir}/agents",
            f"{self.tests_dir}/tools",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def create_agent(
        self,
        user_id: str,
        agent_name: str,
        description: str,
        capabilities: List[str],
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new agent with permission enforcement and template application.

        Args:
            user_id: ID of the user creating the agent
            agent_name: Name of the agent
            description: Description of the agent's purpose
            capabilities: List of agent capabilities
            author: Author name (defaults to user_id)

        Returns:
            Dict containing creation result
        """
        try:
            # Check permission manually
            if not role_manager.check_permission(user_id, Permission.CREATE_AGENT):
                return {
                    "status": "failure",
                    "result": None,
                    "message": f"User {user_id} lacks required permission: {Permission.CREATE_AGENT.value}",
                }

            logger.info(f"Creating agent '{agent_name}' for user {user_id}")

            # Log audit event
            log_audit_event(
                user_id=user_id,
                action="create_agent",
                resource=agent_name,
                details={
                    "description": description,
                    "capabilities": capabilities,
                    "author": author or user_id,
                },
            )

            # Generate agent code from template
            agent_code = AgentTemplate.generate_agent_code(
                agent_name=agent_name,
                description=description,
                capabilities=capabilities,
                author=author if author else user_id,
                user_id=user_id,
            )

            # Generate test code
            test_code = AgentTemplate.generate_test_code(
                agent_name=agent_name, capabilities=capabilities
            )

            # Generate manifest entry
            manifest_entry = AgentTemplate.generate_manifest_entry(
                agent_name=agent_name,
                description=description,
                capabilities=capabilities,
                author=author if author else user_id,
                user_id=user_id,
            )

            # Write agent file
            agent_filename = agent_name.lower().replace(" ", "_").replace("-", "_")
            agent_file_path = os.path.join(self.agents_dir, f"{agent_filename}.py")

            with open(agent_file_path, "w") as f:
                f.write(agent_code)

            # Write test file
            test_file_path = os.path.join(
                self.tests_dir, "agents", f"test_{agent_filename}.py"
            )
            with open(test_file_path, "w") as f:
                f.write(test_code)

            # Update agents manifest
            self._update_agents_manifest(manifest_entry)

            # Run tests
            test_result = self._run_tests(test_file_path)

            # Update documentation
            self._update_agents_documentation(manifest_entry)

            logger.info(f"Successfully created agent '{agent_name}'")

            return {
                "status": "success",
                "result": agent_filename,
                "message": f"Successfully created agent '{agent_name}'",
                "files_created": [agent_file_path, test_file_path],
                "test_result": test_result,
                "manifest_entry": manifest_entry,
            }

        except Exception as e:
            logger.error(f"Failed to create agent '{agent_name}': {e}")
            log_audit_event(
                user_id=user_id,
                action="create_agent_failed",
                resource=agent_name,
                details={"error": str(e)},
            )

            return {
                "status": "failure",
                "result": None,
                "message": f"Failed to create agent: {str(e)}",
            }

    def create_tool(
        self,
        user_id: str,
        tool_name: str,
        description: str,
        parameters: Dict[str, str],
        return_type: str,
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new tool with permission enforcement and template application.

        Args:
            user_id: ID of the user creating the tool
            tool_name: Name of the tool
            description: Description of the tool's purpose
            parameters: Dictionary of parameter names and types
            return_type: Return type of the tool
            author: Author name (defaults to user_id)

        Returns:
            Dict containing creation result
        """
        try:
            # Check permission manually
            if not role_manager.check_permission(user_id, Permission.CREATE_TOOL):
                return {
                    "status": "failure",
                    "result": None,
                    "message": f"User {user_id} lacks required permission: {Permission.CREATE_TOOL.value}",
                }

            logger.info(f"Creating tool '{tool_name}' for user {user_id}")

            # Log audit event
            log_audit_event(
                user_id=user_id,
                action="create_tool",
                resource=tool_name,
                details={
                    "description": description,
                    "parameters": parameters,
                    "return_type": return_type,
                    "author": author or user_id,
                },
            )

            # Generate tool code from template
            tool_code = ToolTemplate.generate_tool_code(
                tool_name=tool_name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                author=author or user_id,
                user_id=user_id,
            )

            # Generate test code
            test_code = ToolTemplate.generate_test_code(
                tool_name=tool_name, parameters=parameters, return_type=return_type
            )

            # Generate manifest entry
            manifest_entry = ToolTemplate.generate_manifest_entry(
                tool_name=tool_name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                author=author or user_id,
                user_id=user_id,
            )

            # Write tool file
            tool_filename = tool_name.lower().replace(" ", "_").replace("-", "_")
            tool_file_path = os.path.join(self.tools_dir, f"{tool_filename}.py")

            with open(tool_file_path, "w") as f:
                f.write(tool_code)

            # Write test file
            test_file_path = os.path.join(
                self.tests_dir, "tools", f"test_{tool_filename}.py"
            )
            with open(test_file_path, "w") as f:
                f.write(test_code)

            # Update tools manifest
            self._update_tools_manifest(manifest_entry)

            # Run tests
            test_result = self._run_tests(test_file_path)

            # Update documentation
            self._update_tools_documentation(manifest_entry)

            logger.info(f"Successfully created tool '{tool_name}'")

            return {
                "status": "success",
                "result": tool_filename,
                "message": f"Successfully created tool '{tool_name}'",
                "files_created": [tool_file_path, test_file_path],
                "test_result": test_result,
                "manifest_entry": manifest_entry,
            }

        except Exception as e:
            logger.error(f"Failed to create tool '{tool_name}': {e}")
            log_audit_event(
                user_id=user_id,
                action="create_tool_failed",
                resource=tool_name,
                details={"error": str(e)},
            )

            return {
                "status": "failure",
                "result": None,
                "message": f"Failed to create tool: {str(e)}",
            }

    def _update_agents_manifest(self, manifest_entry: Dict[str, Any]):
        """Update the agents manifest file"""
        try:
            if os.path.exists(self.agents_manifest):
                with open(self.agents_manifest, "r") as f:
                    manifest_data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(manifest_data, list):
                        existing_agents = manifest_data
                    else:
                        existing_agents = manifest_data.get("agents", [])
            else:
                existing_agents = []

            # Check if agent already exists
            for agent in existing_agents:
                if agent.get("name") == manifest_entry["name"]:
                    # Update existing entry
                    agent.update(manifest_entry)
                    break
            else:
                # Add new entry
                existing_agents.append(manifest_entry)

            # Write back in the same format as the original
            if os.path.exists(self.agents_manifest):
                with open(self.agents_manifest, "r") as f:
                    original_data = json.load(f)
                    if isinstance(original_data, list):
                        # Keep as list format
                        with open(self.agents_manifest, "w") as f:
                            json.dump(existing_agents, f, indent=2)
                    else:
                        # Keep as dict format
                        manifest_data = {"agents": existing_agents}
                        with open(self.agents_manifest, "w") as f:
                            json.dump(manifest_data, f, indent=2)
            else:
                # Default to list format for new files
                with open(self.agents_manifest, "w") as f:
                    json.dump(existing_agents, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update agents manifest: {e}")
            raise

    def _update_tools_manifest(self, manifest_entry: Dict[str, Any]):
        """Update the tools manifest file"""
        try:
            if os.path.exists(self.tools_manifest):
                with open(self.tools_manifest, "r") as f:
                    manifest_data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(manifest_data, list):
                        existing_tools = manifest_data
                    else:
                        existing_tools = manifest_data.get("tools", [])
            else:
                existing_tools = []

            # Check if tool already exists
            for tool in existing_tools:
                if tool.get("name") == manifest_entry["name"]:
                    # Update existing entry
                    tool.update(manifest_entry)
                    break
            else:
                # Add new entry
                existing_tools.append(manifest_entry)

            # Write back in the same format as the original
            if os.path.exists(self.tools_manifest):
                with open(self.tools_manifest, "r") as f:
                    original_data = json.load(f)
                    if isinstance(original_data, list):
                        # Keep as list format
                        with open(self.tools_manifest, "w") as f:
                            json.dump(existing_tools, f, indent=2)
                    else:
                        # Keep as dict format
                        manifest_data = {"tools": existing_tools}
                        with open(self.tools_manifest, "w") as f:
                            json.dump(manifest_data, f, indent=2)
            else:
                # Default to list format for new files
                with open(self.tools_manifest, "w") as f:
                    json.dump(existing_tools, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update tools manifest: {e}")
            raise

    def _run_tests(self, test_file_path: str) -> Dict[str, Any]:
        """Run tests for the created entity"""
        try:
            logger.info(f"Running tests for {test_file_path}")

            # Run the test file
            result = subprocess.run(
                [sys.executable, "-m", "unittest", test_file_path, "-v"],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Test execution timed out",
                "return_code": -1,
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "return_code": -1}

    def _update_agents_documentation(self, manifest_entry: Dict[str, Any]):
        """Update agents documentation"""
        try:
            docs_file = "docs/AGENTS.md"
            os.makedirs("docs", exist_ok=True)

            # Create or update documentation
            if os.path.exists(docs_file):
                with open(docs_file, "r") as f:
                    content = f.read()
            else:
                content = "# Agents Documentation\n\n"

            # Add new agent documentation
            agent_name = manifest_entry["name"]
            description = manifest_entry["description"]
            capabilities = manifest_entry["capabilities"]
            author = manifest_entry["author"]
            created_at = manifest_entry["created_at"]

            agent_doc = f"""
## {agent_name.replace('_', ' ').title()}

**Description:** {description}

**Author:** {author}

**Created:** {created_at}

**Capabilities:**
{chr(10).join([f'- {cap}' for cap in capabilities])}

**File:** `agents/{agent_name}.py`

---
"""

            # Insert new agent documentation
            if f"## {agent_name.replace('_', ' ').title()}" not in content:
                content += agent_doc

            with open(docs_file, "w") as f:
                f.write(content)

        except Exception as e:
            logger.warning(f"Failed to update agents documentation: {e}")

    def _update_tools_documentation(self, manifest_entry: Dict[str, Any]):
        """Update tools documentation"""
        try:
            docs_file = "docs/TOOLS.md"
            os.makedirs("docs", exist_ok=True)

            # Create or update documentation
            if os.path.exists(docs_file):
                with open(docs_file, "r") as f:
                    content = f.read()
            else:
                content = "# Tools Documentation\n\n"

            # Add new tool documentation
            tool_name = manifest_entry["name"]
            description = manifest_entry["description"]
            parameters = manifest_entry["parameters"]
            return_type = manifest_entry["return_type"]
            author = manifest_entry["author"]
            created_at = manifest_entry["created_at"]

            tool_doc = f"""
## {tool_name.replace('_', ' ').title()}

**Description:** {description}

**Author:** {author}

**Created:** {created_at}

**Parameters:**
{chr(10).join([f'- {param} ({param_type})' for param, param_type in parameters.items()])}

**Returns:** {return_type}

**File:** `tools/{tool_name}.py`

---
"""

            # Insert new tool documentation
            if f"## {tool_name.replace('_', ' ').title()}" not in content:
                content += tool_doc

            with open(docs_file, "w") as f:
                f.write(content)

        except Exception as e:
            logger.warning(f"Failed to update tools documentation: {e}")

    def list_entities(self, entity_type: str = "all") -> Dict[str, Any]:
        """List all entities of the specified type"""
        try:
            result = {}

            if entity_type in ["all", "agents"]:
                if os.path.exists(self.agents_manifest):
                    with open(self.agents_manifest, "r") as f:
                        manifest_data = json.load(f)
                        # Handle both list and dict formats
                        if isinstance(manifest_data, list):
                            result["agents"] = manifest_data
                        else:
                            result["agents"] = manifest_data.get("agents", [])
                else:
                    result["agents"] = []

            if entity_type in ["all", "tools"]:
                if os.path.exists(self.tools_manifest):
                    with open(self.tools_manifest, "r") as f:
                        manifest_data = json.load(f)
                        # Handle both list and dict formats
                        if isinstance(manifest_data, list):
                            result["tools"] = manifest_data
                        else:
                            result["tools"] = manifest_data.get("tools", [])
                else:
                    result["tools"] = []

            return {"status": "success", "result": result}

        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return {"status": "failure", "result": None, "message": str(e)}


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Create agents and tools with permission enforcement"
    )
    parser.add_argument(
        "--user-id", required=True, help="User ID for permission checks"
    )
    parser.add_argument(
        "--entity-type",
        choices=["agent", "tool"],
        required=True,
        help="Type of entity to create",
    )
    parser.add_argument("--name", required=True, help="Name of the entity")
    parser.add_argument(
        "--description", required=True, help="Description of the entity"
    )
    parser.add_argument("--author", help="Author name (defaults to user-id)")

    # Agent-specific arguments
    parser.add_argument(
        "--capabilities", nargs="+", help="Agent capabilities (for agents)"
    )

    # Tool-specific arguments
    parser.add_argument("--parameters", help="Tool parameters as JSON (for tools)")
    parser.add_argument("--return-type", help="Tool return type (for tools)")

    # Other options
    parser.add_argument("--list", action="store_true", help="List existing entities")
    parser.add_argument(
        "--list-type",
        choices=["agents", "tools", "all"],
        default="all",
        help="Type of entities to list",
    )

    args = parser.parse_args()

    creator = EntityCreator()

    if args.list:
        # List entities
        result = creator.list_entities(args.list_type)
        if result["status"] == "success":
            print(json.dumps(result["result"], indent=2))
        else:
            print(f"Error: {result['message']}")
            sys.exit(1)
    else:
        # Create entity
        if args.entity_type == "agent":
            if not args.capabilities:
                print("Error: --capabilities is required for agents")
                sys.exit(1)

            result = creator.create_agent(
                user_id=args.user_id,
                agent_name=args.name,
                description=args.description,
                capabilities=args.capabilities,
                author=args.author,
            )
        elif args.entity_type == "tool":
            if not args.parameters or not args.return_type:
                print("Error: --parameters and --return-type are required for tools")
                sys.exit(1)

            try:
                parameters = json.loads(args.parameters)
            except json.JSONDecodeError:
                print("Error: --parameters must be valid JSON")
                sys.exit(1)

            result = creator.create_tool(
                user_id=args.user_id,
                tool_name=args.name,
                description=args.description,
                parameters=parameters,
                return_type=args.return_type,
                author=args.author,
            )

        # Print result
        print(json.dumps(result, indent=2))

        if result["status"] == "failure":
            sys.exit(1)


if __name__ == "__main__":
    main()
