# 📋 User Setup Checklist - Phase 2 Onboarding

## 🎯 Overview

This checklist guides new team members through the complete setup process for the permissioned creation system. Follow each step in order and check off items as you complete them.

**Estimated Time**: 30-45 minutes  
**Prerequisites**: Python 3.11+, Git, basic terminal knowledge  

## 👤 Pre-Setup Information

### Your Details
- **Name**: _________________
- **Username**: _________________
- **Role**: □ Tool Maker □ Agent Smith □ Admin
- **Email**: _________________
- **GitHub Username**: _________________

### System Information
- **Repository URL**: _________________
- **Team Lead**: _________________
- **System Admin**: _________________
- **Support Channel**: #permissioned-system-help

---

## 🔧 Environment Setup

### Step 1: Prerequisites Verification
- [ ] **Python Version Check**
  ```bash
  python --version
  # Should show Python 3.11.x or higher
  ```
  
- [ ] **Git Installation Check**
  ```bash
  git --version
  # Should show git version 2.x or higher
  ```
  
- [ ] **Terminal/Command Line Access**
  - Windows: PowerShell or Command Prompt
  - macOS/Linux: Terminal
  - VS Code: Integrated Terminal

### Step 2: Repository Setup
- [ ] **Clone Repository**
  ```bash
  git clone <repository-url>
  cd <project-directory>
  ```
  
- [ ] **Verify Project Structure**
  ```bash
  ls -la
  # Should show: create_entity.py, core/, templates/, etc.
  ```

### Step 3: Python Environment
- [ ] **Create Virtual Environment**
  ```bash
  python -m venv venv
  ```
  
- [ ] **Activate Virtual Environment**
  ```bash
  # Windows:
  venv\Scripts\activate
  
  # macOS/Linux:
  source venv/bin/activate
  ```
  
- [ ] **Verify Activation**
  ```bash
  which python
  # Should point to your venv directory
  ```

### Step 4: Dependencies Installation
- [ ] **Upgrade pip**
  ```bash
  python -m pip install --upgrade pip
  ```
  
- [ ] **Install Requirements**
  ```bash
  pip install -r requirements.txt
  ```
  
- [ ] **Verify Installation**
  ```bash
  python -c "import langchain_core, langgraph, pydantic; print('✅ Dependencies OK')"
  ```

### Step 5: System Setup
- [ ] **Run Setup Script**
  ```bash
  python setup_permissioned_system.py
  ```
  
- [ ] **Verify Setup**
  ```bash
  python validate_phase1.py
  # Should show all checks passing
  ```

---

## 🔐 User Account Setup

### Step 6: Role Assignment
- [ ] **Contact System Admin**
  - Request user account creation
  - Provide your username and desired role
  - Wait for confirmation email

- [ ] **Verify Account Creation**
  ```bash
  python -c "
  from core.roles import RoleManager
  rm = RoleManager()
  user = rm.get_user('your_username')
  print(f'Role: {user.role}')
  print(f'Permissions: {user.get_permissions()}')
  "
  ```

### Step 7: Git Configuration
- [ ] **Configure Git Identity**
  ```bash
  git config user.name "Your Full Name"
  git config user.email "your.email@company.com"
  ```
  
- [ ] **Verify Git Configuration**
  ```bash
  git config --list
  # Should show your name and email
  ```

### Step 8: Git Hooks Setup
- [ ] **Make Pre-commit Hook Executable**
  ```bash
  chmod +x .git/hooks/pre-commit
  ```
  
- [ ] **Test Git Hook**
  ```bash
  echo "# Test file" > test_setup.py
  git add test_setup.py
  git commit -m "test: setup verification"
  # Should show validation messages
  git reset --hard HEAD~1
  rm test_setup.py
  ```

---

## 🛠️ Development Environment

### Step 9: VS Code Setup
- [ ] **Open Project in VS Code**
  ```bash
  code .
  ```
  
- [ ] **Select Python Interpreter**
  - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
  - Type "Python: Select Interpreter"
  - Choose the interpreter from your venv directory

- [ ] **Install Recommended Extensions**
  - VS Code should prompt for recommended extensions
  - Click "Install All" or install individually:
    - Python
    - Pylance
    - Black Formatter
    - isort
    - Flake8
    - GitLens

- [ ] **Verify VS Code Configuration**
  - Check that `.vscode/settings.json` exists
  - Verify Python interpreter is set correctly
  - Test auto-formatting with `Shift+Alt+F`

### Step 10: Testing Environment
- [ ] **Run System Tests**
  ```bash
  python test_permissioned_system.py
  # Should show all tests passing
  ```
  
- [ ] **Test pytest**
  ```bash
  python -m pytest --version
  # Should show pytest version
  ```

---

## 🧪 First Entity Creation

### Step 11: Tool Creation (Tool Maker Role)
- [ ] **Create Your First Tool**
  ```bash
  python create_entity.py tool my_first_tool \
    --user=your_username \
    --role=tool_maker \
    --description="My first tool for learning" \
    --category="learning" \
    --tags="first,tool,learning"
  ```
  
- [ ] **Verify Tool Creation**
  - Check that `tools/my_first_tool.py` exists
  - Check that `tests/tools/test_my_first_tool.py` exists
  - Check that `docs/tools/my_first_tool.md` exists

- [ ] **Run Tool Tests**
  ```bash
  python -m pytest tests/tools/test_my_first_tool.py -v
  ```

### Step 12: Agent Creation (Agent Smith Role)
- [ ] **Create Your First Agent**
  ```bash
  python create_entity.py agent my_first_agent \
    --user=your_username \
    --role=agent_smith \
    --description="My first agent for learning" \
    --tools="my_first_tool" \
    --category="learning" \
    --tags="first,agent,learning"
  ```
  
- [ ] **Verify Agent Creation**
  - Check that `agents/my_first_agent.py` exists
  - Check that `tests/agents/test_my_first_agent.py` exists
  - Check that `docs/agents/my_first_agent.md` exists

- [ ] **Run Agent Tests**
  ```bash
  python -m pytest tests/agents/test_my_first_agent.py -v
  ```

---

## 📚 Documentation Review

### Step 13: Documentation Access
- [ ] **Review System Documentation**
  - Read `PERMISSIONED_CREATION_SYSTEM.md`
  - Read `docs/ONBOARDING_GUIDE.md`
  - Read `SETUP_INSTRUCTIONS.md`

- [ ] **Bookmark Key Resources**
  - Training schedule: `docs/TRAINING_SCHEDULE.md`
  - Best practices: `docs/BEST_PRACTICES.md`
  - Troubleshooting: `docs/TROUBLESHOOTING.md`

### Step 14: Support Channels
- [ ] **Join Communication Channels**
  - Slack: #permissioned-system
  - Slack: #permissioned-system-help
  - GitHub: Repository access

- [ ] **Save Contact Information**
  - Team Lead: _________________
  - System Admin: _________________
  - QA Lead: _________________

---

## ✅ Final Verification

### Step 15: Complete System Test
- [ ] **Run Full Validation**
  ```bash
  python validate_phase1.py
  # Should show all checks passing
  ```
  
- [ ] **Test Complete Workflow**
  ```bash
  # Create a test entity
  python create_entity.py tool test_tool \
    --user=your_username \
    --role=tool_maker \
    --description="Test tool" \
    --category="test"
  
  # Run tests
  python -m pytest tests/tools/test_test_tool.py
  
  # Format code
  black .
  isort .
  
  # Commit (should trigger validation)
  git add .
  git commit -m "feat: add test tool"
  ```

### Step 16: Documentation Update
- [ ] **Update Your Tool/Agent Documentation**
  - Add usage examples
  - Include troubleshooting tips
  - Review for clarity

- [ ] **Create Learning Notes**
  - Document any issues encountered
  - Note solutions for future reference
  - Share insights with team

---

## 🎯 Success Criteria

### Environment Setup
- [ ] Python 3.11+ installed and working
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] System setup script completed
- [ ] Validation script passes all checks

### User Account
- [ ] User account created with correct role
- [ ] Permissions verified and working
- [ ] Git configured with personal details
- [ ] Pre-commit hooks working correctly

### Development Environment
- [ ] VS Code configured and optimized
- [ ] Python interpreter set correctly
- [ ] Recommended extensions installed
- [ ] Auto-formatting working

### First Entity
- [ ] Successfully created first tool/agent
- [ ] Tests pass without errors
- [ ] Documentation generated and complete
- [ ] Can commit changes successfully

### Knowledge
- [ ] Understand system architecture
- [ ] Know your role and permissions
- [ ] Can navigate project structure
- [ ] Familiar with documentation

---

## 🆘 Troubleshooting

### Common Issues
- **Python version too old**: Install Python 3.11+ from python.org
- **Dependencies fail**: Try `pip install --upgrade pip` first
- **Git hooks not working**: Run `chmod +x .git/hooks/pre-commit`
- **VS Code issues**: Reload window and check Python interpreter
- **Permission errors**: Contact system admin for role assignment

### Getting Help
1. **Check documentation**: Review troubleshooting guides
2. **Ask team**: Use Slack channels for quick questions
3. **Contact admin**: For permission and account issues
4. **Create issue**: Use GitHub Issues for bugs

---

## 📞 Support Contacts

- **System Admin**: _________________ (admin@company.com)
- **Team Lead**: _________________ (lead@company.com)
- **QA Lead**: _________________ (qa@company.com)
- **Slack Help**: #permissioned-system-help
- **GitHub Issues**: [Repository Issues]

---

**Setup Completion Date**: _________________  
**Setup Duration**: _________________  
**Issues Encountered**: _________________  
**Notes**: _________________  

**Setup Status**: □ IN PROGRESS □ COMPLETED □ NEEDS HELP

---

**Next Steps**: 
1. Attend Session 1 of training schedule
2. Complete hands-on exercises
3. Participate in code reviews
4. Contribute to team knowledge 