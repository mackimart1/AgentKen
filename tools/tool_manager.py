"""
Tool Manager Module
Manages tool registration, configuration, and lifecycle
"""


class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool_class):
        """
        Register a new tool with the manager

        Args:
            name: Unique name for the tool
            tool_class: Class implementing the tool
        """
        self.tools[name] = tool_class

    def get_tool(self, name: str):
        """
        Retrieve a registered tool

        Args:
            name: Name of the tool to retrieve

        Returns:
            Instance of the tool class
        """
        return self.tools.get(name)
