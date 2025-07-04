import os

"""
Tool Executor Module
Handles execution of various tools used by agents
"""


class ToolExecutor:
    def __init__(self):
        pass

    def execute(self, tool_name: str, *args, **kwargs):
        """
        Execute a specific tool with given arguments

        Args:
            tool_name: Name of the tool to execute
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Result of the tool execution
        """
        if tool_name == "write_to_file":
            path = kwargs.get("path")
            content = kwargs.get("content")
            if not path or not content:
                return {
                    "status": "error",
                    "message": "Missing path or content for write_to_file",
                }
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Write file
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {
                    "status": "success",
                    "message": f"File written successfully to {path}",
                }
            except Exception as e:
                return {"status": "error", "message": f"File write failed: {str(e)}"}
        elif tool_name == "read_file":
            path = kwargs.get("path")
            if not path:
                return {"status": "error", "message": "Missing path for read_file"}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return {"status": "success", "content": f.read()}
            except Exception as e:
                return {"status": "error", "message": f"File read failed: {str(e)}"}
        else:
            raise NotImplementedError(f"Unsupported tool: {tool_name}")
