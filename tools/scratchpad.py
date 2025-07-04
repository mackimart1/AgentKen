from typing import Optional, Union, List, Dict
from langchain_core.tools import tool

# In-memory storage for the scratchpad
_scratchpad_storage: Dict[str, str] = {}


@tool
def scratchpad(
    action: str, key: Optional[str] = None, value: Optional[str] = None
) -> Union[str, List[str], None, Dict[str, str]]:
    """Manages an in-memory key-value scratchpad.

    Args:
        action (str): The operation: 'write', 'read', 'list_keys', 'clear', 'clear_all'.
        key (Optional[str]): The key for 'write', 'read', 'clear'. Required for these actions.
        value (Optional[str]): The value for 'write'. Required for 'write' action.

    Returns:
        Union[str, List[str], None, Dict[str, str]]: Result from the scratchpad operation.
        - 'write': Returns the stored value (str).
        - 'read': Returns the stored value (str) or None if the key is not found.
        - 'list_keys': Returns a list of existing keys (List[str]).
        - 'clear': Returns a confirmation message (str) if deleted, or error message.
        - 'clear_all': Returns a confirmation message (str).
        - Invalid action or missing args: Returns an error message.
    """
    if action == "write":
        if key is None or value is None:
            return "Error: 'write' action requires both 'key' and 'value' parameters."
        _scratchpad_storage[key] = value
        return value

    elif action == "read":
        if key is None:
            return "Error: 'read' action requires 'key' parameter."
        return _scratchpad_storage.get(key, None)

    elif action == "list_keys":
        return list(_scratchpad_storage.keys())

    elif action == "clear":
        if key is None:
            return "Error: 'clear' action requires 'key' parameter."
        if key in _scratchpad_storage:
            del _scratchpad_storage[key]
            return f"Key '{key}' cleared successfully."
        else:
            return f"Key '{key}' not found."

    elif action == "clear_all":
        _scratchpad_storage.clear()
        return "All scratchpad data cleared successfully."

    else:
        return f"Error: Unknown action '{action}'. Valid actions are: write, read, list_keys, clear, clear_all."


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Test write
    result = scratchpad.invoke({"action": "write", "key": "test", "value": "hello"})
    print(f"Write result: {result}")

    # Test read
    result = scratchpad.invoke({"action": "read", "key": "test"})
    print(f"Read result: {result}")

    # Test list_keys
    result = scratchpad.invoke({"action": "list_keys"})
    print(f"List keys result: {result}")

    # Test clear
    result = scratchpad.invoke({"action": "clear", "key": "test"})
    print(f"Clear result: {result}")

    # Test clear_all
    result = scratchpad.invoke({"action": "clear_all"})
    print(f"Clear all result: {result}")
