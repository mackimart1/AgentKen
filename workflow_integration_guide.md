# Development Workflow Integration Guide

## Overview

This guide outlines how to integrate the Permissioned Agent/Tool Creation System into your development workflow, including Git hooks, CI/CD pipelines, and team onboarding procedures.

## 1. Git Integration

### Pre-commit Hooks

Create `.git/hooks/pre-commit` to validate created entities:

```bash
#!/bin/bash
# Pre-commit hook for permissioned creation system

echo "Running permissioned creation system validation..."

# Check for new agents/tools
NEW_AGENTS=$(git diff --cached --name-only | grep "^agents/.*\.py$")
NEW_TOOLS=$(git diff --cached --name-only | grep "^tools/.*\.py$")

# Validate new agents
for agent in $NEW_AGENTS; do
    echo "Validating agent: $agent"
    python -m pytest "tests/agents/test_$(basename $agent .py).py" -v
    if [ $? -ne 0 ]; then
        echo "❌ Agent validation failed: $agent"
        exit 1
    fi
done

# Validate new tools
for tool in $NEW_TOOLS; do
    echo "Validating tool: $tool"
    python -m pytest "tests/tools/test_$(basename $tool .py).py" -v
    if [ $? -ne 0 ]; then
        echo "❌ Tool validation failed: $tool"
        exit 1
    fi
done

echo "✅ All validations passed!"
```

### Commit Message Templates

Create `.gitmessage` template:

```
# Permissioned Creation System Commit

## Type
- [agent] New agent created
- [tool] New tool created
- [permission] Permission/role changes
- [template] Template updates
- [docs] Documentation updates

## Entity Details
- Name: 
- Created by: 
- User ID: 
- Description: 

## Changes
- 

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] Audit log reviewed

## Related Issues
Closes #
```

Configure Git to use the template:
```bash
git config commit.template .gitmessage
```

## 2. CI/CD Pipeline Integration

### GitHub Actions Workflow

Create `.github/workflows/permissioned-system.yml`:

```yaml
name: Permissioned Creation System CI

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'agents/**'
      - 'tools/**'
      - 'tests/**'
      - 'core/roles.py'
      - 'create_entity.py'
  pull_request:
    branches: [ main, develop ]

jobs:
  validate-entities:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run permissioned system tests
      run: |
        python test_permissioned_system.py
    
    - name: Run entity tests
      run: |
        python -m pytest tests/agents/ -v --cov=agents
        python -m pytest tests/tools/ -v --cov=tools
    
    - name: Validate manifests
      run: |
        python -c "
        import json
        with open('agents_manifest.json') as f:
            agents = json.load(f)
        with open('tools_manifest.json') as f:
            tools = json.load(f)
        print(f'Validated {len(agents[\"agents\"])} agents and {len(tools[\"tools\"])} tools')
        "
    
    - name: Check audit logs
      run: |
        if [ -f audit.log ]; then
          echo "Audit log entries:"
          tail -10 audit.log
        fi

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Security scan
      run: |
        # Add security scanning tools
        pip install bandit safety
        bandit -r agents/ tools/ core/
        safety check
```

### GitLab CI Pipeline

Create `.gitlab-ci.yml`:

```yaml
stages:
  - validate
  - test
  - security
  - deploy

validate_entities:
  stage: validate
  script:
    - python test_permissioned_system.py
    - python -c "import json; json.load(open('agents_manifest.json')); json.load(open('tools_manifest.json'))"
  only:
    - main
    - develop

test_entities:
  stage: test
  script:
    - pip install pytest pytest-cov
    - python -m pytest tests/agents/ --cov=agents --cov-report=xml
    - python -m pytest tests/tools/ --cov=tools --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security_scan:
  stage: security
  script:
    - pip install bandit safety
    - bandit -r agents/ tools/ core/ -f json -o bandit-report.json
    - safety check
  artifacts:
    reports:
      sast: bandit-report.json
```

## 3. Development Environment Setup

### Docker Development Environment

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  inferra-dev:
    build: .
    volumes:
      - .:/app
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=development
    ports:
      - "8000:8000"
    command: python -m pytest tests/ --watch
    
  permissioned-system:
    build: .
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
    command: python setup_permissioned_system.py
    depends_on:
      - inferra-dev
```

### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "files.associations": {
    "*.py": "python"
  },
  "emmet.includeLanguages": {
    "python": "html"
  }
}
```

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Create Agent",
      "type": "shell",
      "command": "python",
      "args": ["create_entity.py", "--entity-type", "agent"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Create Tool",
      "type": "shell",
      "command": "python",
      "args": ["create_entity.py", "--entity-type", "tool"],
      "group": "build"
    },
    {
      "label": "Test Permissioned System",
      "type": "shell",
      "command": "python",
      "args": ["test_permissioned_system.py"],
      "group": "test"
    }
  ]
}
```

## 4. Team Onboarding Process

### Onboarding Checklist

Create `onboarding/checklist.md`:

```markdown
# Team Onboarding Checklist

## Pre-Onboarding
- [ ] User account created in permissioned system
- [ ] Role assigned (agent_smith, tool_maker, etc.)
- [ ] Access to development environment granted
- [ ] Git repository access configured

## Development Environment Setup
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Permissioned system initialized (`python setup_permissioned_system.py`)
- [ ] VS Code extensions installed (Python, GitLens, etc.)
- [ ] Git hooks configured

## Training
- [ ] Permissioned system overview (30 min)
- [ ] Agent creation tutorial (45 min)
- [ ] Tool creation tutorial (45 min)
- [ ] Testing and validation (30 min)
- [ ] Git workflow and CI/CD (30 min)

## First Tasks
- [ ] Create a simple test agent
- [ ] Create a simple test tool
- [ ] Submit a pull request
- [ ] Code review process
- [ ] Deploy to development environment

## Documentation Review
- [ ] README_PERMISSIONED_SYSTEM.md
- [ ] PERMISSIONED_CREATION_SYSTEM.md
- [ ] API documentation
- [ ] Troubleshooting guide
```

### Training Materials

Create `onboarding/training/`:

```
onboarding/training/
├── 01-system-overview.md
├── 02-agent-creation.md
├── 03-tool-creation.md
├── 04-testing-validation.md
├── 05-git-workflow.md
├── exercises/
│   ├── create-simple-agent.md
│   ├── create-simple-tool.md
│   └── troubleshooting.md
└── slides/
    ├── system-overview.pptx
    ├── hands-on-demo.pptx
    └── best-practices.pptx
```

## 5. Monitoring and Analytics

### Metrics Collection

Create `monitoring/metrics.py`:

```python
"""
Metrics collection for permissioned creation system
"""
import time
import json
from datetime import datetime
from typing import Dict, Any

class CreationMetrics:
    def __init__(self, metrics_file: str = "metrics/creation_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "agents_created": 0,
                "tools_created": 0,
                "creation_time_avg": 0,
                "success_rate": 1.0,
                "user_activity": {},
                "daily_stats": {}
            }
    
    def record_creation(self, entity_type: str, user_id: str, 
                       success: bool, duration: float):
        """Record a creation event"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update counts
        if entity_type == "agent":
            self.metrics["agents_created"] += 1 if success else 0
        elif entity_type == "tool":
            self.metrics["tools_created"] += 1 if success else 0
        
        # Update user activity
        if user_id not in self.metrics["user_activity"]:
            self.metrics["user_activity"][user_id] = {
                "agents_created": 0,
                "tools_created": 0,
                "last_activity": None
            }
        
        if success:
            if entity_type == "agent":
                self.metrics["user_activity"][user_id]["agents_created"] += 1
            elif entity_type == "tool":
                self.metrics["user_activity"][user_id]["tools_created"] += 1
        
        self.metrics["user_activity"][user_id]["last_activity"] = datetime.now().isoformat()
        
        # Update daily stats
        if today not in self.metrics["daily_stats"]:
            self.metrics["daily_stats"][today] = {
                "agents_created": 0,
                "tools_created": 0,
                "total_creations": 0,
                "successful_creations": 0
            }
        
        self.metrics["daily_stats"][today]["total_creations"] += 1
        if success:
            self.metrics["daily_stats"][today]["successful_creations"] += 1
            if entity_type == "agent":
                self.metrics["daily_stats"][today]["agents_created"] += 1
            elif entity_type == "tool":
                self.metrics["daily_stats"][today]["tools_created"] += 1
        
        # Update averages
        total_creations = sum(day["total_creations"] for day in self.metrics["daily_stats"].values())
        total_successful = sum(day["successful_creations"] for day in self.metrics["daily_stats"].values())
        self.metrics["success_rate"] = total_successful / total_creations if total_creations > 0 else 1.0
        
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard"""
        return {
            "total_agents": self.metrics["agents_created"],
            "total_tools": self.metrics["tools_created"],
            "success_rate": self.metrics["success_rate"],
            "active_users": len([u for u in self.metrics["user_activity"].values() 
                               if u["last_activity"] and 
                               (datetime.now() - datetime.fromisoformat(u["last_activity"])).days < 7]),
            "recent_activity": self.metrics["daily_stats"]
        }
```

## 6. Integration with Existing Systems

### Web API Integration

Update `web_api.py` to include permissioned creation endpoints:

```python
# Add to existing web_api.py
from create_entity import EntityCreator
from core.roles import role_manager, Permission

class PermissionedCreationAPI:
    def __init__(self):
        self.creator = EntityCreator()
    
    def create_agent_endpoint(self, request_data):
        """API endpoint for agent creation"""
        user_id = request_data.get('user_id')
        if not user_id:
            return {"error": "user_id required"}, 400
        
        if not role_manager.check_permission(user_id, Permission.CREATE_AGENT):
            return {"error": "Insufficient permissions"}, 403
        
        result = self.creator.create_agent(
            user_id=user_id,
            agent_name=request_data['name'],
            description=request_data['description'],
            capabilities=request_data['capabilities'],
            author=request_data.get('author', user_id)
        )
        
        return result, 200 if result['status'] == 'success' else 400
    
    def create_tool_endpoint(self, request_data):
        """API endpoint for tool creation"""
        user_id = request_data.get('user_id')
        if not user_id:
            return {"error": "user_id required"}, 400
        
        if not role_manager.check_permission(user_id, Permission.CREATE_TOOL):
            return {"error": "Insufficient permissions"}, 403
        
        result = self.creator.create_tool(
            user_id=user_id,
            tool_name=request_data['name'],
            description=request_data['description'],
            parameters=request_data['parameters'],
            return_type=request_data['return_type'],
            author=request_data.get('author', user_id)
        )
        
        return result, 200 if result['status'] == 'success' else 400
```

## 7. Best Practices and Guidelines

### Code Review Checklist

Create `.github/pull_request_template.md`:

```markdown
## Permissioned Creation System PR

### Type of Change
- [ ] New agent
- [ ] New tool
- [ ] Permission/role changes
- [ ] Template updates
- [ ] Documentation updates
- [ ] Bug fix

### Entity Details
- **Name**: 
- **Created by**: 
- **User ID**: 
- **Description**: 

### Changes Made
- 

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Documentation updated
- [ ] Audit log reviewed

### Security
- [ ] Permission checks implemented
- [ ] Input validation added
- [ ] No sensitive data exposed
- [ ] Audit logging in place

### Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Error handling implemented
- [ ] Performance considered

### Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

### Related Issues
Closes #
```

This integration guide provides a comprehensive framework for incorporating the permissioned creation system into your development workflow, ensuring consistency, security, and quality across all agent and tool creation activities. 