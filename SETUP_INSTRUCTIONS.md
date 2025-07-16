# 🔧 Setup Instructions - Permissioned Creation System

This guide provides step-by-step instructions for setting up and validating the permissioned creation system.

## 📋 Prerequisites

Before running the setup, ensure you have:

- **Python 3.11 or higher**
- **Git** installed and configured
- **pip** (Python package installer)
- **Basic terminal/command line knowledge**

### Check Your Environment

```bash
# Check Python version
python --version
# Should output: Python 3.11.x or higher

# Check pip
pip --version
# Should output: pip 21.x or higher

# Check Git
git --version
# Should output: git version 2.x or higher
```

## 🚀 Step-by-Step Setup

### Step 1: Clone and Navigate

```bash
# Clone the repository (replace with your actual repo URL)
git clone <your-repository-url>
cd <project-directory>

# Verify you're in the correct directory
ls -la
# Should show files like: create_entity.py, core/, templates/, etc.
```

### Step 2: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import langchain_core, langgraph; print('✅ Dependencies installed successfully')"
```

### Step 3: Run the Setup Script

```bash
# Run the main setup script
python setup_permissioned_system.py

# Expected output:
# 🔧 Setting up permissioned creation system...
# 📁 Creating directories...
# 📝 Creating manifest files...
# 📚 Creating documentation...
# 👥 Creating default users...
# ✅ Setup completed successfully!
```

### Step 4: Verify Installation

```bash
# Run the validation script
python test_permissioned_system.py

# Expected output:
# 🧪 Running permissioned system tests...
# ✅ User management tests passed
# ✅ Permission tests passed
# ✅ Role tests passed
# ✅ Template tests passed
# ✅ Creation workflow tests passed
# 🎉 All tests passed!
```

### Step 5: Setup Git Hooks

```bash
# Make pre-commit hook executable
chmod +x .git/hooks/pre-commit

# Test the hook
git add .
git commit -m "Initial setup" --allow-empty
# Should show validation messages and complete successfully
```

## 🔍 Validation Checklist

Run through this checklist to ensure everything is working correctly:

### ✅ Environment Validation

```bash
# 1. Check Python environment
python --version  # Should be 3.11+
which python     # Should point to your virtual environment

# 2. Check dependencies
python -c "
import sys
required_packages = ['langchain_core', 'langgraph', 'pydantic', 'typing_extensions']
missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')
if missing:
    print(f'\\nMissing packages: {missing}')
    sys.exit(1)
else:
    print('\\n🎉 All dependencies installed!')
"

# 3. Check project structure
python -c "
import os
required_dirs = ['agents', 'tools', 'tests', 'tests/agents', 'tests/tools', 'core', 'templates', 'docs', 'docs/agents', 'docs/tools']
required_files = ['create_entity.py', 'core/roles.py', 'templates/agent_template.py', 'templates/tool_template.py']

missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_dirs:
    print(f'❌ Missing directories: {missing_dirs}')
if missing_files:
    print(f'❌ Missing files: {missing_files}')
if not missing_dirs and not missing_files:
    print('✅ Project structure is correct!')
"
```

### ✅ System Validation

```bash
# 4. Test role management
python -c "
from core.roles import RoleManager, User
rm = RoleManager()

# Test user creation
rm.create_user('test_user', 'tool_maker')
user = rm.get_user('test_user')
print(f'✅ User created: {user.username} with role {user.role}')

# Test permissions
permissions = user.get_permissions()
print(f'✅ Permissions: {permissions}')

# Cleanup
rm.delete_user('test_user')
print('✅ User management working correctly')
"

# 5. Test template system
python -c "
from templates.agent_template import AgentTemplate
from templates.tool_template import ToolTemplate

agent_template = AgentTemplate()
tool_template = ToolTemplate()

print('✅ Agent template loaded')
print('✅ Tool template loaded')
"

# 6. Test creation workflow
python -c "
import sys
sys.path.insert(0, '.')
from create_entity import EntityCreator

creator = EntityCreator()
print('✅ Entity creator initialized')
"
```

### ✅ Git Integration Validation

```bash
# 7. Test Git hooks
echo "# Test file" > test_validation.py
git add test_validation.py
git commit -m "Testing pre-commit hook"
# Should show validation messages

# Cleanup test file
git reset --hard HEAD~1
rm -f test_validation.py
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: Python Version Too Old
```
SyntaxError: invalid syntax
```

**Solution**:
```bash
# Check your Python version
python --version

# If below 3.11, install Python 3.11+ from python.org
# Or use pyenv to manage Python versions
```

#### Issue 2: Missing Dependencies
```
ModuleNotFoundError: No module named 'langchain_core'
```

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install langchain-core langgraph pydantic typing-extensions pytest
```

#### Issue 3: Permission Denied
```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# On Windows, run as administrator
# On macOS/Linux, check file permissions:
ls -la .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

#### Issue 4: Git Hook Not Working
```
pre-commit hook failed
```

**Solution**:
```bash
# Check if hook is executable
ls -la .git/hooks/pre-commit

# Make executable
chmod +x .git/hooks/pre-commit

# Test manually
.git/hooks/pre-commit
```

#### Issue 5: Template Files Missing
```
FileNotFoundError: Template file not found
```

**Solution**:
```bash
# Re-run setup
python setup_permissioned_system.py

# Check if templates directory exists
ls -la templates/
```

### Getting Help

If you encounter issues not covered above:

1. **Check the logs**: Look for error messages in the terminal output
2. **Review documentation**: Check `PERMISSIONED_CREATION_SYSTEM.md`
3. **Run validation**: Execute `python test_permissioned_system.py`
4. **Check GitHub Issues**: Look for similar problems
5. **Contact the team**: Reach out to your system administrator

## 🎯 Post-Setup Configuration

### Optional: Configure Your IDE

#### VS Code Settings
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration
1. Open project in PyCharm
2. Go to Settings → Project → Python Interpreter
3. Add interpreter: `./venv/bin/python`
4. Install packages from `requirements.txt`

### Optional: Environment Variables

Create `.env` file for local configuration:
```bash
# Development settings
DEBUG=true
LOG_LEVEL=INFO
AUDIT_LOG_PATH=./audit_log.json

# Optional: External authentication
FIREBASE_PROJECT_ID=your-project-id
CLERK_SECRET_KEY=your-secret-key
```

## 🚀 Next Steps

After successful setup:

1. **Read the onboarding guide**: `docs/ONBOARDING_GUIDE.md`
2. **Create your first entity**: Follow the examples in the onboarding guide
3. **Join team communication**: Connect to Slack/Teams channels
4. **Set up your development workflow**: Configure your IDE and Git
5. **Start contributing**: Begin creating tools or agents based on your role

## 📞 Support

If you need help with setup:

- **Documentation**: Check the docs folder
- **Team Lead**: Contact your team lead for role assignment
- **System Admin**: For technical issues and permissions
- **GitHub Issues**: For bug reports and feature requests

---

**Happy coding! 🎉**

Remember: A successful setup is the foundation for productive development. Take the time to ensure everything is working correctly before starting your first project. 