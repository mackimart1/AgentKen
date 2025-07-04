# Permissioned Agent/Tool Creation System

## Overview

The Permissioned Agent/Tool Creation System provides a secure, automated, and templated approach to creating new agents and tools in the Inferra V system. It enforces role-based permissions, applies consistent templates, runs automated tests, and maintains comprehensive documentation.

## Architecture

### Core Components

1. **Role Management (`core/roles.py`)**
   - Defines user roles and permissions
   - Provides decorators for permission enforcement
   - Manages user-role mappings
   - Handles audit logging

2. **Template System (`templates/`)**
   - `agent_template.py`: Templates for new agents
   - `tool_template.py`: Templates for new tools
   - Enforces consistent code structure and documentation

3. **Centralized Creation (`create_entity.py`)**
   - Unified interface for creating agents and tools
   - Enforces permissions and applies templates
   - Runs automated tests and updates documentation
   - Provides CLI and programmatic interfaces

### User Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| `admin` | All permissions | Full system access |
| `agent_smith` | Create/modify agents, view agents/tools | Can create and modify agents |
| `tool_maker` | Create/modify tools, view agents/tools | Can create and modify tools |
| `viewer` | View agents/tools | Read-only access |
| `guest` | No permissions | Limited access |

### Permissions

- `CREATE_AGENT`: Create new agents
- `CREATE_TOOL`: Create new tools
- `MODIFY_AGENT`: Modify existing agents
- `MODIFY_TOOL`: Modify existing tools
- `DELETE_AGENT`: Delete agents
- `DELETE_TOOL`: Delete tools
- `VIEW_AGENTS`: View agent information
- `VIEW_TOOLS`: View tool information
- `MANAGE_USERS`: Manage user roles
- `VIEW_AUDIT_LOGS`: View audit logs

## Usage

### Command Line Interface

#### Creating an Agent

```bash
python create_entity.py \
  --user-id "john_doe" \
  --entity-type agent \
  --name "Data Analyzer" \
  --description "Analyzes data and generates insights" \
  --capabilities "data_analysis" "statistics" "visualization" \
  --author "John Doe"
```

#### Creating a Tool

```bash
python create_entity.py \
  --user-id "jane_smith" \
  --entity-type tool \
  --name "Data Validator" \
  --description "Validates data format and content" \
  --parameters '{"data": "str", "format": "str"}' \
  --return-type "bool" \
  --author "Jane Smith"
```

#### Listing Entities

```bash
# List all entities
python create_entity.py --user-id "admin" --list

# List only agents
python create_entity.py --user-id "admin" --list --list-type agents

# List only tools
python create_entity.py --user-id "admin" --list --list-type tools
```

### Programmatic Interface

#### Creating an Agent

```python
from create_entity import EntityCreator

creator = EntityCreator()

result = creator.create_agent(
    user_id="john_doe",
    agent_name="Data Analyzer",
    description="Analyzes data and generates insights",
    capabilities=["data_analysis", "statistics", "visualization"],
    author="John Doe"
)

print(result)
```

#### Creating a Tool

```python
from create_entity import EntityCreator

creator = EntityCreator()

result = creator.create_tool(
    user_id="jane_smith",
    tool_name="Data Validator",
    description="Validates data format and content",
    parameters={"data": "str", "format": "str"},
    return_type="bool",
    author="Jane Smith"
)

print(result)
```

## User Management

### Setting User Roles

```python
from core.roles import role_manager, UserRole

# Set a user as an agent smith
role_manager.set_user_role("john_doe", UserRole.AGENT_SMITH)

# Set a user as a tool maker
role_manager.set_user_role("jane_smith", UserRole.TOOL_MAKER)

# Set a user as admin
role_manager.set_user_role("admin_user", UserRole.ADMIN)
```

### Checking Permissions

```python
from core.roles import role_manager, Permission

# Check if user can create agents
can_create_agent = role_manager.check_permission("john_doe", Permission.CREATE_AGENT)

# Get all user permissions
permissions = role_manager.get_user_permissions("john_doe")
```

### Using Permission Decorators

```python
from core.roles import requires_permission, requires_role, Permission, UserRole

@requires_permission(Permission.CREATE_AGENT)
def create_custom_agent(user_id: str, **kwargs):
    # This function can only be called by users with CREATE_AGENT permission
    pass

@requires_role(UserRole.AGENT_SMITH)
def agent_smith_only_function(user_id: str, **kwargs):
    # This function can only be called by agent_smith users
    pass
```

## Template System

### Agent Templates

The agent template system generates:

1. **Agent Code** (`agents/agent_name.py`)
   - Standard LangGraph workflow structure
   - Proper error handling and logging
   - Memory integration
   - Tool integration
   - Comprehensive documentation

2. **Test Code** (`tests/agents/test_agent_name.py`)
   - Unit tests for all agent functionality
   - Mock testing for dependencies
   - Error condition testing
   - Integration testing

3. **Manifest Entry** (in `agents_manifest.json`)
   - Agent metadata
   - Capabilities list
   - Creation information
   - Version tracking

### Tool Templates

The tool template system generates:

1. **Tool Code** (`tools/tool_name.py`)
   - LangChain tool decorator
   - Parameter validation
   - Error handling
   - Logging
   - Type hints

2. **Test Code** (`tests/tools/test_tool_name.py`)
   - Parameter validation tests
   - Error handling tests
   - Return type tests
   - Integration tests

3. **Manifest Entry** (in `tools_manifest.json`)
   - Tool metadata
   - Parameter definitions
   - Return type information
   - Creation information

## File Structure

```
project_root/
├── core/
│   └── roles.py                 # Role and permission management
├── templates/
│   ├── agent_template.py        # Agent templates
│   └── tool_template.py         # Tool templates
├── create_entity.py             # Centralized creation system
├── agents/                      # Generated agents
├── tools/                       # Generated tools
├── tests/                       # Generated tests
│   ├── agents/
│   └── tools/
├── docs/                        # Auto-generated documentation
│   ├── AGENTS.md
│   └── TOOLS.md
├── agents_manifest.json         # Agent registry
├── tools_manifest.json          # Tool registry
├── users.json                   # User-role mappings
├── roles.json                   # Role-permission mappings
└── audit.log                    # Audit trail
```

## Security Features

### Permission Enforcement

- All creation operations require appropriate permissions
- Role-based access control at function level
- Decorator-based permission checking
- Automatic permission validation

### Audit Logging

- All creation events are logged
- User actions are tracked with timestamps
- Failed operations are logged with error details
- Audit trail is maintained in `audit.log`

### Input Validation

- Template-generated code includes parameter validation
- Type checking for all inputs
- Error handling for invalid inputs
- Comprehensive test coverage

## Best Practices

### For Agent Creation

1. **Clear Capabilities**: Define specific, measurable capabilities
2. **Proper Documentation**: Use descriptive names and detailed descriptions
3. **Error Handling**: Ensure robust error handling in generated code
4. **Testing**: Write comprehensive tests for all functionality
5. **Memory Integration**: Leverage the memory system for context

### For Tool Creation

1. **Parameter Design**: Use clear, descriptive parameter names
2. **Type Safety**: Specify proper types for all parameters
3. **Validation**: Include comprehensive input validation
4. **Error Messages**: Provide clear, actionable error messages
5. **Documentation**: Document all parameters and return values

### For User Management

1. **Principle of Least Privilege**: Grant only necessary permissions
2. **Regular Review**: Periodically review user roles and permissions
3. **Audit Monitoring**: Monitor audit logs for suspicious activity
4. **Role Separation**: Separate agent and tool creation roles

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Check user role: `role_manager.get_user_role(user_id)`
   - Verify permissions: `role_manager.check_permission(user_id, permission)`
   - Ensure user exists in `users.json`

2. **Template Generation Errors**
   - Check parameter types and names
   - Ensure all required fields are provided
   - Verify file paths and permissions

3. **Test Failures**
   - Review generated test code
   - Check for missing dependencies
   - Verify test environment setup

4. **Manifest Update Errors**
   - Check file permissions
   - Verify JSON format
   - Ensure backup of existing manifests

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Override

For emergency situations, you can temporarily bypass permissions:

```python
# WARNING: Only use in emergencies
role_manager.set_user_role("emergency_user", UserRole.ADMIN)
```

## Migration Guide

### From Manual Creation

1. **Backup Existing Code**: Create backups of existing agents/tools
2. **Update User Roles**: Set appropriate roles for existing users
3. **Migrate Manifests**: Update existing manifest files to new format
4. **Test System**: Verify all functionality works with new system
5. **Update Documentation**: Regenerate documentation using new system

### From Legacy System

1. **Install New Dependencies**: Ensure all required packages are installed
2. **Initialize Roles**: Set up user roles and permissions
3. **Migrate Entities**: Use the creation system to recreate existing entities
4. **Update References**: Update any code that references old entity locations
5. **Verify Integration**: Test integration with existing system components

## Future Enhancements

### Planned Features

1. **Web Interface**: Web-based creation interface
2. **Version Control**: Git integration for entity versioning
3. **Approval Workflow**: Multi-step approval process for entity creation
4. **Templates Library**: Community-contributed templates
5. **Performance Monitoring**: Metrics and performance tracking
6. **API Rate Limiting**: Rate limiting for creation operations

### Extension Points

1. **Custom Templates**: Support for custom template types
2. **Plugin System**: Plugin architecture for additional functionality
3. **External Integrations**: Integration with external systems
4. **Advanced Permissions**: Fine-grained permission controls
5. **Workflow Automation**: Automated workflows for complex entity creation

## Support

For issues and questions:

1. Check the audit logs for error details
2. Review the troubleshooting section
3. Enable debug logging for detailed information
4. Consult the API documentation
5. Contact the system administrator

## License

This system is part of the Inferra V project and follows the same licensing terms. 