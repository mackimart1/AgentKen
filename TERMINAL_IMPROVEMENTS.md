# Terminal Command Improvements for AgentK

This document outlines the enhanced terminal command capabilities added to the AgentK project.

## Overview

The project now includes two powerful tools for executing terminal commands:

1. **`run_shell_command`** - Enhanced single command execution with security and timeout features
2. **`terminal_session`** - Advanced persistent terminal sessions with state management

## 1. Enhanced Shell Command Tool (`run_shell_command`)

### Features
- **Security**: Blocks dangerous commands that could harm the system
- **Timeout Support**: Configurable command timeout (default: 30 seconds)
- **Working Directory**: Execute commands in specific directories
- **Detailed Output**: Comprehensive result information including execution time
- **Error Handling**: Robust error handling with clear error messages

### Usage Examples

```python
from tools.run_shell_command import run_shell_command

# Basic command execution
result = run_shell_command.invoke({"command": "echo 'Hello World'"})
print(f"Status: {result['status']}")
print(f"Output: {result['stdout']}")
print(f"Execution time: {result['execution_time']}s")

# Command with custom timeout and working directory
result = run_shell_command.invoke({
    "command": "dir",
    "timeout": 60,
    "working_directory": "C:\\temp"
})

# Command that will be blocked for security
result = run_shell_command.invoke({"command": "rm -rf /"})
print(f"Blocked: {result['message']}")  # Will show security warning
```

### Return Format
```json
{
    "status": "success|failure",
    "stdout": "command output",
    "stderr": "error output",
    "returncode": 0,
    "message": "human-readable status",
    "execution_time": 1.23
}
```

## 2. Advanced Terminal Session Tool (`terminal_session`)

### Features
- **Persistent Sessions**: Maintain state across multiple commands
- **Working Directory Tracking**: Automatic directory change tracking
- **Environment Variables**: Set and maintain environment variables per session
- **Command History**: Track all executed commands with timestamps
- **Multi-Command Execution**: Execute multiple commands in sequence
- **Session Management**: Create, list, destroy, and inspect sessions

### Usage Examples

#### Basic Session Management
```python
from tools.terminal_session import terminal_session

# Create a new session
result = terminal_session.invoke({
    "action": "create",
    "session_id": "my_session",
    "working_directory": "/home/user/project",
    "environment_vars": {"NODE_ENV": "development"}
})

# List all sessions
result = terminal_session.invoke({"action": "list"})
print(f"Active sessions: {result['data']['session_count']}")

# Get session information
result = terminal_session.invoke({
    "action": "info",
    "session_id": "my_session"
})
```

#### Command Execution
```python
# Execute a single command
result = terminal_session.invoke({
    "action": "execute",
    "session_id": "my_session",
    "command": "npm install"
})

# Change directory (persistent across commands)
terminal_session.invoke({
    "action": "execute",
    "session_id": "my_session",
    "command": "cd src"
})

# Set environment variable (persistent in session)
terminal_session.invoke({
    "action": "execute",
    "session_id": "my_session",
    "command": "DEBUG=true"
})

# Execute multiple commands
multi_commands = """
npm run build
npm test
npm run deploy
"""

result = terminal_session.invoke({
    "action": "execute",
    "session_id": "my_session",
    "command": multi_commands,
    "multi_command": True
})
```

#### Session History and Cleanup
```python
# View command history
result = terminal_session.invoke({
    "action": "history",
    "session_id": "my_session"
})

for cmd in result['data']['history']:
    print(f"{cmd['timestamp']}: {cmd['command']} -> {cmd['result']['status']}")

# Destroy session when done
terminal_session.invoke({
    "action": "destroy",
    "session_id": "my_session"
})
```

### Session Actions
- **`create`**: Create a new terminal session
- **`execute`**: Execute commands in a session
- **`list`**: List all active sessions
- **`destroy`**: Remove a session
- **`info`**: Get detailed session information
- **`history`**: View command history for a session

## Security Features

Both tools include security measures to prevent dangerous operations:

### Blocked Command Patterns
- `rm -rf /` - Recursive deletion of root directory
- `format c:` - Format system drive
- `mkfs` - Create filesystem (can destroy data)
- `:(){ :|:& };:` - Fork bomb
- `sudo rm -rf` - Privileged recursive deletion
- `dd if=/dev/zero` - Disk wiping operations

### Additional Safety Features
- Input validation and sanitization
- Timeout protection against hanging commands
- Working directory validation
- Environment variable isolation per session

## Integration with AgentK

These tools are now registered in the `tools_manifest.json` and can be used by any agent in the AgentK system:

- **Agent Smith** uses `run_shell_command` for code formatting, linting, and testing
- **Software Engineer** can use both tools for development workflows
- **Tool Maker** uses `run_shell_command` for testing created tools
- Any custom agents can leverage these tools for system operations

## Testing

Comprehensive test suites are included:
- `tests/tools/test_run_shell_command.py` - Tests for the enhanced shell command tool
- `tests/tools/test_terminal_session.py` - Tests for the terminal session tool

Run tests with:
```bash
python -m pytest tests/tools/test_run_shell_command.py -v
python -m pytest tests/tools/test_terminal_session.py -v
```

## Benefits for AgentK

1. **Enhanced Automation**: Agents can now perform complex multi-step terminal operations
2. **Better State Management**: Persistent sessions allow for stateful operations
3. **Improved Security**: Built-in protections against dangerous commands
4. **Better Debugging**: Detailed execution information and command history
5. **Flexibility**: Support for both simple commands and complex workflows

## Future Enhancements

Potential future improvements:
- Interactive command support (stdin handling)
- Real-time output streaming
- Command output filtering and parsing
- Integration with container environments
- Remote command execution capabilities
- Command templates and macros

---

These terminal improvements significantly enhance AgentK's ability to interact with the operating system and perform complex automation tasks safely and efficiently.