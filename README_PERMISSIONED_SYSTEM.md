# Permissioned Agent/Tool Creation System

This directory contains the permissioned creation system for agents and tools.

## Quick Start

1. **Set up users and roles:**
   ```python
   from core.roles import role_manager, UserRole
   
   # Create an agent smith
   role_manager.set_user_role("john_doe", UserRole.AGENT_SMITH)
   
   # Create a tool maker  
   role_manager.set_user_role("jane_smith", UserRole.TOOL_MAKER)
   ```

2. **Create an agent:**
   ```bash
   python create_entity.py \
     --user-id "john_doe" \
     --entity-type agent \
     --name "Data Analyzer" \
     --description "Analyzes data and generates insights" \
     --capabilities "data_analysis" "statistics" "visualization"
   ```

3. **Create a tool:**
   ```bash
   python create_entity.py \
     --user-id "jane_smith" \
     --entity-type tool \
     --name "Data Validator" \
     --description "Validates data format and content" \
     --parameters '{"data": "str", "format": "str"}' \
     --return-type "bool"
   ```

## Testing

Run the test script to verify the system:

```bash
python test_permissioned_system.py
```

## Documentation

- [System Documentation](PERMISSIONED_CREATION_SYSTEM.md)
- [Agents Documentation](docs/AGENTS.md)
- [Tools Documentation](docs/TOOLS.md)

## Files

- `core/roles.py` - Role and permission management
- `templates/` - Code templates for agents and tools
- `create_entity.py` - Centralized creation system
- `test_permissioned_system.py` - Test script
- `setup_permissioned_system.py` - Setup script (this file)

## Default Users

The system creates these default users:

- `admin` - Full system access
- `agent_smith_example` - Can create agents
- `tool_maker_example` - Can create tools  
- `viewer_example` - Read-only access

## Security

- All operations require appropriate permissions
- Audit logging tracks all actions
- Role-based access control enforced
- Input validation on all parameters

For more information, see the main documentation file.
