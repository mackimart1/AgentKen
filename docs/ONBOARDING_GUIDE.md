# 🚀 Onboarding Guide - Permissioned Creation System

Welcome to the team! This guide will help you get started with our permissioned creation system for agents and tools.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [User Roles & Permissions](#user-roles--permissions)
4. [Getting Started](#getting-started)
5. [Creating Your First Entity](#creating-your-first-entity)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Resources](#resources)

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.11 or higher
- Git
- Basic understanding of Python and LangChain/LangGraph

### Initial Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Run the setup script
python setup_permissioned_system.py

# Verify installation
python test_permissioned_system.py
```

## 🏗️ System Overview

Our permissioned creation system ensures consistent, high-quality agent and tool development through:

- **Role-based access control** - Different permissions for different team members
- **Automated templates** - Consistent code structure and documentation
- **Quality gates** - Automated testing and validation
- **Audit logging** - Track all changes and who made them

### Key Components

```
project/
├── core/roles.py              # Role and permission management
├── create_entity.py           # Main creation interface
├── templates/                 # Code templates
│   ├── agent_template.py
│   └── tool_template.py
├── agents/                    # Generated agents
├── tools/                     # Generated tools
├── tests/                     # Generated tests
└── docs/                      # Generated documentation
```

## 👥 User Roles & Permissions

### Tool Maker Role
**Purpose**: Create and maintain tools that agents can use

**Permissions**:
- ✅ Create new tools
- ✅ Modify existing tools (if created by them)
- ✅ View all tools
- ✅ Run tool tests
- ❌ Create agents
- ❌ Modify agents

**Best for**: Backend developers, API specialists, data engineers

### Agent Smith Role
**Purpose**: Create and maintain agents that use tools

**Permissions**:
- ✅ Create new agents
- ✅ Modify existing agents (if created by them)
- ✅ View all agents and tools
- ✅ Run agent tests
- ❌ Create tools
- ❌ Modify tools

**Best for**: AI/ML engineers, prompt engineers, system architects

### Admin Role
**Purpose**: Full system management

**Permissions**:
- ✅ All Tool Maker permissions
- ✅ All Agent Smith permissions
- ✅ Manage user roles
- ✅ View audit logs
- ✅ Override permissions
- ✅ System configuration

**Best for**: Team leads, system administrators

## 🎯 Getting Started

### 1. Account Setup

Your team lead will create your account with the appropriate role:

```python
from core.roles import User, RoleManager

# Admin creates your account
rm = RoleManager()
rm.create_user("your_username", "tool_maker")  # or "agent_smith"
```

### 2. Environment Verification

Run these commands to verify your setup:

```bash
# Check Python version
python --version  # Should be 3.11+

# Check dependencies
python -c "import langchain_core, langgraph; print('✅ Dependencies OK')"

# Test permissioned system
python test_permissioned_system.py

# Check your role
python -c "
from core.roles import User, RoleManager
rm = RoleManager()
user = rm.get_user('your_username')
print(f'Role: {user.role}')
print(f'Permissions: {user.get_permissions()}')
"
```

### 3. Git Hooks Setup

The system includes pre-commit hooks that automatically validate your code:

```bash
# Make sure hooks are executable
chmod +x .git/hooks/pre-commit

# Test the hook
git add .
git commit -m "test"  # This will run validation
```

## 🛠️ Creating Your First Entity

### Creating a Tool (Tool Maker)

```bash
# Create a simple calculator tool
python create_entity.py tool calculator \
  --user=your_username \
  --role=tool_maker \
  --description="Basic arithmetic operations" \
  --category="math" \
  --tags="calculator,math,arithmetic"
```

This will create:
- `tools/calculator.py` - Your tool implementation
- `tests/tools/test_calculator.py` - Unit tests
- `docs/tools/calculator.md` - Documentation
- Updates to `tools_manifest.json`

### Creating an Agent (Agent Smith)

```bash
# Create a math agent that uses the calculator
python create_entity.py agent math_agent \
  --user=your_username \
  --role=agent_smith \
  --description="Agent that performs mathematical operations" \
  --tools="calculator" \
  --category="math" \
  --tags="math,calculator,agent"
```

This will create:
- `agents/math_agent.py` - Your agent implementation
- `tests/agents/test_math_agent.py` - Unit tests
- `docs/agents/math_agent.md` - Documentation
- Updates to `agents_manifest.json`

## 📝 Best Practices

### Code Quality

1. **Follow the templates** - Don't modify the generated structure
2. **Write comprehensive tests** - Aim for 90%+ coverage
3. **Document everything** - Clear docstrings and README files
4. **Use type hints** - All functions should have proper typing

### Naming Conventions

- **Tools**: Use descriptive, lowercase names (`calculator`, `weather_api`)
- **Agents**: Use descriptive names with `_agent` suffix (`math_agent`, `customer_service_agent`)
- **Files**: Follow Python naming conventions (snake_case)

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest tests/tools/test_calculator.py -v

# Run with coverage
python -m pytest --cov=tools --cov-report=html
```

### Documentation

Every entity should have:
- Clear docstring explaining purpose
- Usage examples
- Input/output specifications
- Error handling documentation

## 🔧 Troubleshooting

### Common Issues

#### Permission Denied
```
PermissionError: User 'username' does not have permission to create agents
```

**Solution**: Contact your admin to assign the correct role.

#### Template Not Found
```
FileNotFoundError: Template file not found
```

**Solution**: Run `python setup_permissioned_system.py` to recreate templates.

#### Test Failures
```
AssertionError: Test failed
```

**Solution**: 
1. Check your implementation against the template
2. Verify all required imports are present
3. Ensure your function signatures match the template

#### Git Hook Fails
```
pre-commit hook failed
```

**Solution**:
1. Fix any linting issues: `black .` and `isort .`
2. Ensure all tests pass
3. Check manifest file syntax

### Getting Help

1. **Check the logs**: Look at `audit_log.json` for detailed error information
2. **Review documentation**: Check `PERMISSIONED_CREATION_SYSTEM.md`
3. **Ask the team**: Use your team's communication channels
4. **Check GitHub Issues**: Look for similar problems

## 📚 Resources

### Documentation
- [System Architecture](PERMISSIONED_CREATION_SYSTEM.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Workflow Integration](workflow_integration_guide.md)

### Code Examples
- [Agent Examples](docs/agents/)
- [Tool Examples](docs/tools/)
- [Test Examples](tests/)

### External Resources
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Python Testing with pytest](https://docs.pytest.org/)

## 🎓 Training Exercises

### Exercise 1: Create a Simple Tool
Create a tool that converts temperatures between Celsius and Fahrenheit.

**Steps**:
1. Use the tool creation command
2. Implement the conversion logic
3. Write tests for both conversions
4. Verify the tool works correctly

### Exercise 2: Create an Agent
Create an agent that uses your temperature conversion tool.

**Steps**:
1. Use the agent creation command
2. Implement the agent logic
3. Write tests for the agent
4. Test the full workflow

### Exercise 3: Debug and Fix
Intentionally introduce a bug and practice debugging.

**Steps**:
1. Create a tool with a syntax error
2. Try to commit (should fail)
3. Fix the error
4. Verify the fix works

## 🏆 Success Metrics

You're ready to contribute when you can:

- ✅ Create tools and agents without errors
- ✅ Write tests that pass consistently
- ✅ Follow the coding standards
- ✅ Use Git hooks effectively
- ✅ Debug common issues independently
- ✅ Help other team members

## 📞 Support

### Team Contacts
- **System Admin**: [Admin Name] - [admin@company.com]
- **Tool Maker Lead**: [Lead Name] - [lead@company.com]
- **Agent Smith Lead**: [Lead Name] - [lead@company.com]

### Communication Channels
- **Slack**: #permissioned-system
- **Email**: permissioned-system@company.com
- **GitHub Issues**: [Repository Issues]

---

**Welcome to the team! 🎉**

Remember: The goal is to create high-quality, maintainable code that helps the entire team succeed. Don't hesitate to ask questions and contribute to improving the system! 