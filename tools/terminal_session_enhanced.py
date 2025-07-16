"""
Enhanced Terminal Session Tool with Session Templates and Multi-Agent Collaboration

Key Enhancements:
1. Session Templates: Pre-configured templates for common workflows
2. Multi-Agent Collaboration: Support for cooperative workflows and real-time debugging

This enhanced version provides enterprise-grade terminal session management with
collaboration features and workflow automation.
"""

import subprocess
import os
import tempfile
import json
import time
import threading
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Types of terminal sessions."""
    STANDARD = "standard"
    TEMPLATE = "template"
    COLLABORATIVE = "collaborative"


class AgentRole(Enum):
    """Roles for agents in collaborative sessions."""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    OBSERVER = "observer"
    ADMIN = "admin"


class TemplateCategory(Enum):
    """Categories for session templates."""
    DEVELOPMENT = "development"
    DEPLOYMENT = "deployment"
    DEBUGGING = "debugging"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    ANALYSIS = "analysis"


@dataclass
class AgentParticipant:
    """Represents an agent participating in a collaborative session."""
    agent_id: str
    agent_name: str
    role: AgentRole
    joined_at: datetime
    last_activity: datetime
    permissions: List[str] = field(default_factory=list)
    active: bool = True
    
    def update_activity(self):
        self.last_activity = datetime.now()


@dataclass
class SessionTemplate:
    """Represents a pre-configured session template."""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    commands: List[str]
    environment_vars: Dict[str, str]
    working_directory: Optional[str]
    prerequisites: List[str]
    expected_duration: int  # minutes
    created_by: str
    created_at: datetime
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "commands": self.commands,
            "environment_vars": self.environment_vars,
            "working_directory": self.working_directory,
            "prerequisites": self.prerequisites,
            "expected_duration": self.expected_duration,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
            "tags": self.tags
        }


@dataclass
class CollaborationEvent:
    """Represents an event in a collaborative session."""
    event_id: str
    session_id: str
    agent_id: str
    event_type: str  # command, join, leave, message, lock, unlock
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class EnhancedTerminalSession:
    """Enhanced terminal session with templates and collaboration support."""
    session_id: str
    session_type: SessionType = SessionType.STANDARD
    working_directory: str = field(default_factory=os.getcwd)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    command_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    
    # Template-related fields
    template_id: Optional[str] = None
    template_progress: int = 0  # Current step in template
    
    # Collaboration-related fields
    participants: Dict[str, AgentParticipant] = field(default_factory=dict)
    collaboration_events: List[CollaborationEvent] = field(default_factory=list)
    locked_by: Optional[str] = None
    locked_at: Optional[datetime] = None
    shared_state: Dict[str, Any] = field(default_factory=dict)
    
    # Session metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    max_participants: int = 10
    
    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used = datetime.now()
    
    def add_to_history(self, command: str, result: Dict[str, Any], agent_id: str = "system"):
        """Add a command and its result to the history."""
        self.command_history.append({
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "result": result,
            "agent_id": agent_id
        })
        # Keep only last 200 commands
        if len(self.command_history) > 200:
            self.command_history = self.command_history[-200:]
    
    def add_participant(self, agent_id: str, agent_name: str, role: AgentRole) -> bool:
        """Add a participant to the collaborative session."""
        if len(self.participants) >= self.max_participants:
            return False
        
        if agent_id in self.participants:
            # Update existing participant
            self.participants[agent_id].active = True
            self.participants[agent_id].update_activity()
        else:
            # Add new participant
            self.participants[agent_id] = AgentParticipant(
                agent_id=agent_id,
                agent_name=agent_name,
                role=role,
                joined_at=datetime.now(),
                last_activity=datetime.now()
            )
        
        # Add collaboration event
        self.add_collaboration_event(
            agent_id=agent_id,
            event_type="join",
            content=f"Agent {agent_name} joined as {role.value}"
        )
        
        return True
    
    def remove_participant(self, agent_id: str) -> bool:
        """Remove a participant from the collaborative session."""
        if agent_id not in self.participants:
            return False
        
        participant = self.participants[agent_id]
        participant.active = False
        
        # Add collaboration event
        self.add_collaboration_event(
            agent_id=agent_id,
            event_type="leave",
            content=f"Agent {participant.agent_name} left the session"
        )
        
        # Release lock if held by this agent
        if self.locked_by == agent_id:
            self.unlock_session()
        
        return True
    
    def lock_session(self, agent_id: str) -> bool:
        """Lock the session for exclusive access."""
        if self.locked_by and self.locked_by != agent_id:
            return False
        
        self.locked_by = agent_id
        self.locked_at = datetime.now()
        
        self.add_collaboration_event(
            agent_id=agent_id,
            event_type="lock",
            content="Session locked for exclusive access"
        )
        
        return True
    
    def unlock_session(self) -> bool:
        """Unlock the session."""
        if not self.locked_by:
            return False
        
        agent_id = self.locked_by
        self.locked_by = None
        self.locked_at = None
        
        self.add_collaboration_event(
            agent_id=agent_id,
            event_type="unlock",
            content="Session unlocked"
        )
        
        return True
    
    def add_collaboration_event(self, agent_id: str, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add an event to the collaboration log."""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            agent_id=agent_id,
            event_type=event_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.collaboration_events.append(event)
        
        # Keep only last 500 events
        if len(self.collaboration_events) > 500:
            self.collaboration_events = self.collaboration_events[-500:]
    
    def can_execute(self, agent_id: str) -> bool:
        """Check if an agent can execute commands in this session."""
        if self.session_type == SessionType.STANDARD:
            return True
        
        if agent_id not in self.participants:
            return False
        
        participant = self.participants[agent_id]
        if not participant.active:
            return False
        
        if self.locked_by and self.locked_by != agent_id:
            return False
        
        # Check role permissions
        if participant.role in [AgentRole.OWNER, AgentRole.ADMIN, AgentRole.COLLABORATOR]:
            return True
        
        return False


class SessionTemplateManager:
    """Manages session templates for common workflows."""
    
    def __init__(self):
        self.templates: Dict[str, SessionTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default session templates."""
        default_templates = [
            {
                "name": "Git Setup",
                "description": "Initialize Git repository and configure basic settings",
                "category": TemplateCategory.DEVELOPMENT,
                "commands": [
                    "git init",
                    "git config user.name 'Agent'",
                    "git config user.email 'agent@agentk.ai'",
                    "echo '# Project' > README.md",
                    "echo '.env\n*.log\n__pycache__/\n.vscode/' > .gitignore",
                    "git add .",
                    "git commit -m 'Initial commit'"
                ],
                "environment_vars": {"GIT_EDITOR": "nano"},
                "prerequisites": ["Git must be installed"],
                "expected_duration": 5,
                "tags": ["git", "version-control", "setup"]
            },
            {
                "name": "Python Environment",
                "description": "Set up Python virtual environment and install dependencies",
                "category": TemplateCategory.DEVELOPMENT,
                "commands": [
                    "python -m venv venv",
                    "source venv/bin/activate || venv\\Scripts\\activate",
                    "python -m pip install --upgrade pip",
                    "pip install pytest black flake8 mypy",
                    "echo 'pytest\nblack\nflake8\nmypy' > requirements-dev.txt",
                    "pip freeze > requirements.txt"
                ],
                "environment_vars": {"PYTHONPATH": "."},
                "prerequisites": ["Python 3.7+ must be installed"],
                "expected_duration": 10,
                "tags": ["python", "virtual-environment", "development"]
            },
            {
                "name": "Docker Deploy",
                "description": "Build and deploy Docker container",
                "category": TemplateCategory.DEPLOYMENT,
                "commands": [
                    "docker build -t app:latest .",
                    "docker run --name app-container -d -p 8080:8080 app:latest",
                    "docker ps",
                    "docker logs app-container"
                ],
                "environment_vars": {"DOCKER_BUILDKIT": "1"},
                "prerequisites": ["Docker must be installed", "Dockerfile must exist"],
                "expected_duration": 15,
                "tags": ["docker", "deployment", "containerization"]
            },
            {
                "name": "Node.js Setup",
                "description": "Initialize Node.js project and install dependencies",
                "category": TemplateCategory.DEVELOPMENT,
                "commands": [
                    "npm init -y",
                    "npm install express",
                    "npm install --save-dev nodemon jest eslint",
                    "mkdir src tests",
                    "echo 'console.log(\"Hello World!\");' > src/index.js"
                ],
                "environment_vars": {"NODE_ENV": "development"},
                "prerequisites": ["Node.js and npm must be installed"],
                "expected_duration": 8,
                "tags": ["nodejs", "javascript", "setup"]
            },
            {
                "name": "System Diagnostics",
                "description": "Run comprehensive system diagnostics",
                "category": TemplateCategory.DEBUGGING,
                "commands": [
                    "echo '=== System Information ==='",
                    "uname -a || systeminfo",
                    "echo '=== Disk Usage ==='",
                    "df -h || dir",
                    "echo '=== Memory Usage ==='",
                    "free -h || wmic OS get TotalVisibleMemorySize,FreePhysicalMemory",
                    "echo '=== Process List ==='",
                    "ps aux || tasklist",
                    "echo '=== Network Status ==='",
                    "netstat -tuln"
                ],
                "environment_vars": {},
                "prerequisites": ["System administration access"],
                "expected_duration": 5,
                "tags": ["diagnostics", "system", "debugging"]
            },
            {
                "name": "Database Backup",
                "description": "Create database backup and verify integrity",
                "category": TemplateCategory.MAINTENANCE,
                "commands": [
                    "echo 'Starting database backup...'",
                    "mkdir -p backups/$(date +%Y%m%d)",
                    "# Add your database backup commands here",
                    "echo 'Backup completed successfully'"
                ],
                "environment_vars": {"BACKUP_DIR": "./backups"},
                "prerequisites": ["Database access credentials"],
                "expected_duration": 20,
                "tags": ["database", "backup", "maintenance"]
            },
            {
                "name": "Log Analysis",
                "description": "Analyze system and application logs",
                "category": TemplateCategory.ANALYSIS,
                "commands": [
                    "echo '=== Recent System Logs ==='",
                    "tail -n 50 /var/log/syslog || echo 'Syslog not available'",
                    "echo '=== Error Patterns ==='",
                    "grep -i error /var/log/* 2>/dev/null | tail -n 20 || echo 'No error logs found'",
                    "echo '=== Disk Space Analysis ==='",
                    "du -sh /* 2>/dev/null | sort -hr | head -n 10"
                ],
                "environment_vars": {},
                "prerequisites": ["Log file access permissions"],
                "expected_duration": 10,
                "tags": ["logs", "analysis", "troubleshooting"]
            },
            {
                "name": "Security Audit",
                "description": "Perform basic security audit checks",
                "category": TemplateCategory.ANALYSIS,
                "commands": [
                    "echo '=== Open Ports ==='",
                    "netstat -tuln",
                    "echo '=== Failed Login Attempts ==='",
                    "grep 'Failed password' /var/log/auth.log 2>/dev/null | tail -n 10 || echo 'No auth logs available'",
                    "echo '=== File Permissions Check ==='",
                    "find /etc -type f -perm -002 2>/dev/null | head -n 10 || echo 'Permission check completed'",
                    "echo '=== Process Analysis ==='",
                    "ps aux --sort=-%cpu | head -n 10"
                ],
                "environment_vars": {},
                "prerequisites": ["System administration access"],
                "expected_duration": 15,
                "tags": ["security", "audit", "analysis"]
            }
        ]
        
        for template_data in default_templates:
            template_id = self._generate_template_id(template_data["name"])
            template = SessionTemplate(
                template_id=template_id,
                name=template_data["name"],
                description=template_data["description"],
                category=template_data["category"],
                commands=template_data["commands"],
                environment_vars=template_data["environment_vars"],
                working_directory=None,
                prerequisites=template_data["prerequisites"],
                expected_duration=template_data["expected_duration"],
                created_by="system",
                created_at=datetime.now(),
                tags=template_data["tags"]
            )
            self.templates[template_id] = template
    
    def _generate_template_id(self, name: str) -> str:
        """Generate a unique template ID."""
        return hashlib.md5(name.encode()).hexdigest()[:8]
    
    def create_template(self, name: str, description: str, category: TemplateCategory,
                       commands: List[str], environment_vars: Dict[str, str] = None,
                       working_directory: str = None, prerequisites: List[str] = None,
                       expected_duration: int = 30, created_by: str = "user",
                       tags: List[str] = None) -> str:
        """Create a new session template."""
        template_id = self._generate_template_id(f"{name}_{datetime.now().isoformat()}")
        
        template = SessionTemplate(
            template_id=template_id,
            name=name,
            description=description,
            category=category,
            commands=commands,
            environment_vars=environment_vars or {},
            working_directory=working_directory,
            prerequisites=prerequisites or [],
            expected_duration=expected_duration,
            created_by=created_by,
            created_at=datetime.now(),
            tags=tags or []
        )
        
        self.templates[template_id] = template
        return template_id
    
    def get_template(self, template_id: str) -> Optional[SessionTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[TemplateCategory] = None,
                      tags: List[str] = None) -> List[SessionTemplate]:
        """List templates with optional filtering."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return sorted(templates, key=lambda t: t.usage_count, reverse=True)
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False


class EnhancedTerminalSessionManager:
    """Enhanced terminal session manager with templates and collaboration."""
    
    def __init__(self):
        self.sessions: Dict[str, EnhancedTerminalSession] = {}
        self.template_manager = SessionTemplateManager()
        self._lock = threading.RLock()
    
    def create_session(self, session_id: str, session_type: SessionType = SessionType.STANDARD,
                      working_directory: str = None, environment_vars: Dict[str, str] = None,
                      template_id: str = None, agent_id: str = "system",
                      agent_name: str = "System", description: str = "",
                      max_participants: int = 10) -> Dict[str, Any]:
        """Create a new enhanced terminal session."""
        with self._lock:
            if session_id in self.sessions:
                return {
                    "status": "failure",
                    "message": f"Session '{session_id}' already exists"
                }
            
            # Validate working directory
            if working_directory and not os.path.exists(working_directory):
                return {
                    "status": "failure",
                    "message": f"Working directory does not exist: {working_directory}"
                }
            
            # Create session
            session = EnhancedTerminalSession(
                session_id=session_id,
                session_type=session_type,
                working_directory=working_directory or os.getcwd(),
                environment_vars=environment_vars or {},
                template_id=template_id,
                description=description,
                max_participants=max_participants
            )
            
            # Apply template if specified
            if template_id:
                template = self.template_manager.get_template(template_id)
                if template:
                    session.environment_vars.update(template.environment_vars)
                    if template.working_directory:
                        session.working_directory = template.working_directory
                    session.tags.extend(template.tags)
                    template.usage_count += 1
                else:
                    return {
                        "status": "failure",
                        "message": f"Template '{template_id}' not found"
                    }
            
            # Add creator as participant for collaborative sessions
            if session_type == SessionType.COLLABORATIVE:
                session.add_participant(agent_id, agent_name, AgentRole.OWNER)
            
            self.sessions[session_id] = session
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "session_type": session_type.value,
                    "working_directory": session.working_directory,
                    "environment_vars": session.environment_vars,
                    "template_id": template_id,
                    "created_at": session.created_at.isoformat()
                },
                "message": f"Enhanced terminal session '{session_id}' created successfully"
            }
    
    def execute_command(self, session_id: str, command: str, agent_id: str = "system",
                       timeout: int = 30, auto_advance_template: bool = True) -> Dict[str, Any]:
        """Execute a command in an enhanced session."""
        with self._lock:
            if session_id not in self.sessions:
                # Auto-create session
                create_result = self.create_session(session_id, agent_id=agent_id)
                if create_result["status"] != "success":
                    return create_result
            
            session = self.sessions[session_id]
            
            # Check execution permissions
            if not session.can_execute(agent_id):
                return {
                    "status": "failure",
                    "message": f"Agent '{agent_id}' does not have permission to execute commands in this session"
                }
            
            session.update_last_used()
            
            # Update participant activity
            if agent_id in session.participants:
                session.participants[agent_id].update_activity()
            
            # Execute command
            result = self._execute_single_command(session, command, timeout)
            
            # Add to history with agent info
            session.add_to_history(command, result["data"], agent_id)
            
            # Add collaboration event
            if session.session_type == SessionType.COLLABORATIVE:
                session.add_collaboration_event(
                    agent_id=agent_id,
                    event_type="command",
                    content=command,
                    metadata={"status": result["status"], "returncode": result["data"].get("returncode")}
                )
            
            # Auto-advance template if applicable
            if auto_advance_template and session.template_id:
                template = self.template_manager.get_template(session.template_id)
                if template and session.template_progress < len(template.commands):
                    if command.strip() == template.commands[session.template_progress].strip():
                        session.template_progress += 1
            
            return result
    
    def execute_template(self, session_id: str, template_id: str, agent_id: str = "system",
                        timeout: int = 30, interactive: bool = False) -> Dict[str, Any]:
        """Execute all commands from a template."""
        template = self.template_manager.get_template(template_id)
        if not template:
            return {
                "status": "failure",
                "message": f"Template '{template_id}' not found"
            }
        
        # Create session with template
        if session_id not in self.sessions:
            create_result = self.create_session(
                session_id=session_id,
                session_type=SessionType.TEMPLATE,
                template_id=template_id,
                agent_id=agent_id,
                description=f"Template execution: {template.name}"
            )
            if create_result["status"] != "success":
                return create_result
        
        session = self.sessions[session_id]
        results = []
        
        for i, command in enumerate(template.commands):
            if interactive:
                # In interactive mode, return next command to execute
                if i == session.template_progress:
                    return {
                        "status": "success",
                        "data": {
                            "next_command": command,
                            "progress": f"{i + 1}/{len(template.commands)}",
                            "template_name": template.name
                        },
                        "message": f"Next command in template: {command}"
                    }
            else:
                # Execute command automatically
                result = self.execute_command(session_id, command, agent_id, timeout, False)
                results.append({
                    "command": command,
                    "result": result
                })
                
                session.template_progress = i + 1
                
                # Stop on failure for critical commands
                if result["status"] == "failure" and result["data"]["returncode"] != 0:
                    break
        
        if not interactive:
            return {
                "status": "success",
                "data": {
                    "template_name": template.name,
                    "commands_executed": len(results),
                    "results": results,
                    "completed": session.template_progress >= len(template.commands)
                },
                "message": f"Template '{template.name}' execution completed"
            }
        
        return {
            "status": "success",
            "data": {"message": "Template execution completed"},
            "message": "All template commands have been executed"
        }
    
    def join_session(self, session_id: str, agent_id: str, agent_name: str,
                    role: AgentRole = AgentRole.COLLABORATOR) -> Dict[str, Any]:
        """Join a collaborative session."""
        with self._lock:
            if session_id not in self.sessions:
                return {
                    "status": "failure",
                    "message": f"Session '{session_id}' not found"
                }
            
            session = self.sessions[session_id]
            
            if session.session_type != SessionType.COLLABORATIVE:
                return {
                    "status": "failure",
                    "message": "Session is not collaborative"
                }
            
            if not session.add_participant(agent_id, agent_name, role):
                return {
                    "status": "failure",
                    "message": "Unable to join session (may be full)"
                }
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "role": role.value,
                    "participants": len([p for p in session.participants.values() if p.active])
                },
                "message": f"Agent '{agent_name}' joined collaborative session"
            }
    
    def leave_session(self, session_id: str, agent_id: str) -> Dict[str, Any]:
        """Leave a collaborative session."""
        with self._lock:
            if session_id not in self.sessions:
                return {
                    "status": "failure",
                    "message": f"Session '{session_id}' not found"
                }
            
            session = self.sessions[session_id]
            
            if not session.remove_participant(agent_id):
                return {
                    "status": "failure",
                    "message": f"Agent '{agent_id}' is not a participant"
                }
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "agent_id": agent_id
                },
                "message": "Left collaborative session"
            }
    
    def lock_session(self, session_id: str, agent_id: str) -> Dict[str, Any]:
        """Lock a session for exclusive access."""
        with self._lock:
            if session_id not in self.sessions:
                return {
                    "status": "failure",
                    "message": f"Session '{session_id}' not found"
                }
            
            session = self.sessions[session_id]
            
            if not session.lock_session(agent_id):
                return {
                    "status": "failure",
                    "message": f"Session is already locked by '{session.locked_by}'"
                }
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "locked_by": agent_id,
                    "locked_at": session.locked_at.isoformat()
                },
                "message": "Session locked for exclusive access"
            }
    
    def unlock_session(self, session_id: str, agent_id: str) -> Dict[str, Any]:
        """Unlock a session."""
        with self._lock:
            if session_id not in self.sessions:
                return {
                    "status": "failure",
                    "message": f"Session '{session_id}' not found"
                }
            
            session = self.sessions[session_id]
            
            if session.locked_by != agent_id:
                return {
                    "status": "failure",
                    "message": "Session is not locked by this agent"
                }
            
            session.unlock_session()
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "unlocked_by": agent_id
                },
                "message": "Session unlocked"
            }
    
    def get_collaboration_events(self, session_id: str, agent_id: str = None,
                               limit: int = 50) -> Dict[str, Any]:
        """Get collaboration events for a session."""
        if session_id not in self.sessions:
            return {
                "status": "failure",
                "message": f"Session '{session_id}' not found"
            }
        
        session = self.sessions[session_id]
        events = session.collaboration_events
        
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        
        events = events[-limit:] if limit > 0 else events
        
        return {
            "status": "success",
            "data": {
                "session_id": session_id,
                "events": [e.to_dict() for e in events],
                "total_events": len(session.collaboration_events),
                "showing": len(events)
            },
            "message": f"Retrieved {len(events)} collaboration events"
        }
    
    def _execute_single_command(self, session: EnhancedTerminalSession, command: str, timeout: int) -> Dict[str, Any]:
        """Execute a single command in the session."""
        start_time = time.time()
        
        try:
            # Security check
            dangerous_patterns = [
                "rm -rf /", "del /f /s /q C:\\", "format c:", "mkfs",
                ":(){ :|:& };:", "sudo rm -rf", "dd if=/dev/zero"
            ]
            
            command_lower = command.lower()
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    return {
                        "status": "failure",
                        "data": {
                            "status": "failure",
                            "stdout": "",
                            "stderr": f"Command blocked for security: contains dangerous pattern '{pattern}'",
                            "returncode": -1,
                            "execution_time": 0,
                            "working_directory": session.working_directory
                        },
                        "message": "Command blocked for security reasons"
                    }
            
            # Prepare environment
            env = os.environ.copy()
            env.update(session.environment_vars)
            
            # Handle directory changes
            if command.strip().startswith("cd "):
                new_dir = command.strip()[3:].strip()
                if new_dir:
                    if not os.path.isabs(new_dir):
                        new_dir = os.path.join(session.working_directory, new_dir)
                    new_dir = os.path.abspath(new_dir)
                    
                    if os.path.exists(new_dir) and os.path.isdir(new_dir):
                        session.working_directory = new_dir
                        result_data = {
                            "status": "success",
                            "stdout": f"Changed directory to: {new_dir}",
                            "stderr": "",
                            "returncode": 0,
                            "execution_time": time.time() - start_time,
                            "working_directory": new_dir
                        }
                    else:
                        result_data = {
                            "status": "failure",
                            "stdout": "",
                            "stderr": f"Directory not found: {new_dir}",
                            "returncode": 1,
                            "execution_time": time.time() - start_time,
                            "working_directory": session.working_directory
                        }
                else:
                    home_dir = os.path.expanduser("~")
                    session.working_directory = home_dir
                    result_data = {
                        "status": "success",
                        "stdout": f"Changed directory to: {home_dir}",
                        "stderr": "",
                        "returncode": 0,
                        "execution_time": time.time() - start_time,
                        "working_directory": home_dir
                    }
            
            # Handle environment variable setting
            elif "=" in command and not command.strip().startswith(("echo", "printf", "test", "[")):
                if " " not in command.strip():
                    var_name, var_value = command.strip().split("=", 1)
                    session.environment_vars[var_name] = var_value
                    result_data = {
                        "status": "success",
                        "stdout": f"Set environment variable: {var_name}={var_value}",
                        "stderr": "",
                        "returncode": 0,
                        "execution_time": time.time() - start_time,
                        "working_directory": session.working_directory
                    }
                else:
                    result_data = self._run_subprocess_command(command, session, env, timeout, start_time)
            else:
                result_data = self._run_subprocess_command(command, session, env, timeout, start_time)
            
            return {
                "status": result_data["status"],
                "data": result_data,
                "message": f"Command executed in session '{session.session_id}'"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "status": "failure",
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "execution_time": execution_time,
                "working_directory": session.working_directory
            }
            
            return {
                "status": "failure",
                "data": error_result,
                "message": f"Error executing command: {str(e)}"
            }
    
    def _run_subprocess_command(self, command: str, session: EnhancedTerminalSession,
                               env: Dict[str, str], timeout: int, start_time: float) -> Dict[str, Any]:
        """Run a command using subprocess."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=session.working_directory,
                env=env
            )
            
            execution_time = time.time() - start_time
            status = "success" if result.returncode == 0 else "failure"
            
            return {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time": execution_time,
                "working_directory": session.working_directory
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failure",
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1,
                "execution_time": time.time() - start_time,
                "working_directory": session.working_directory
            }


# Global enhanced session manager
_enhanced_session_manager = EnhancedTerminalSessionManager()


# Input models for the enhanced tools
class CreateSessionInput(BaseModel):
    session_id: str = Field(description="Unique session identifier")
    session_type: str = Field(default="standard", description="Session type: standard, template, collaborative")
    working_directory: Optional[str] = Field(default=None, description="Initial working directory")
    environment_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    template_id: Optional[str] = Field(default=None, description="Template ID to apply")
    agent_id: str = Field(default="system", description="Agent creating the session")
    agent_name: str = Field(default="System", description="Agent name")
    description: str = Field(default="", description="Session description")
    max_participants: int = Field(default=10, description="Maximum participants for collaborative sessions")


class ExecuteCommandInput(BaseModel):
    session_id: str = Field(description="Session identifier")
    command: str = Field(description="Command to execute")
    agent_id: str = Field(default="system", description="Agent executing the command")
    timeout: int = Field(default=30, description="Command timeout in seconds")
    auto_advance_template: bool = Field(default=True, description="Auto-advance template progress")


class ExecuteTemplateInput(BaseModel):
    session_id: str = Field(description="Session identifier")
    template_id: str = Field(description="Template ID to execute")
    agent_id: str = Field(default="system", description="Agent executing the template")
    timeout: int = Field(default=30, description="Command timeout in seconds")
    interactive: bool = Field(default=False, description="Interactive template execution")


class JoinSessionInput(BaseModel):
    session_id: str = Field(description="Session identifier")
    agent_id: str = Field(description="Agent joining the session")
    agent_name: str = Field(description="Agent name")
    role: str = Field(default="collaborator", description="Agent role: owner, collaborator, observer, admin")


class SessionLockInput(BaseModel):
    session_id: str = Field(description="Session identifier")
    agent_id: str = Field(description="Agent requesting lock/unlock")


class CreateTemplateInput(BaseModel):
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    category: str = Field(description="Template category")
    commands: List[str] = Field(description="List of commands")
    environment_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    working_directory: Optional[str] = Field(default=None, description="Working directory")
    prerequisites: Optional[List[str]] = Field(default=None, description="Prerequisites")
    expected_duration: int = Field(default=30, description="Expected duration in minutes")
    created_by: str = Field(default="user", description="Creator identifier")
    tags: Optional[List[str]] = Field(default=None, description="Template tags")


class ListTemplatesInput(BaseModel):
    category: Optional[str] = Field(default=None, description="Filter by category")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")


# Enhanced terminal session tools
@tool(args_schema=CreateSessionInput)
def terminal_session_create_enhanced(
    session_id: str,
    session_type: str = "standard",
    working_directory: Optional[str] = None,
    environment_vars: Optional[Dict[str, str]] = None,
    template_id: Optional[str] = None,
    agent_id: str = "system",
    agent_name: str = "System",
    description: str = "",
    max_participants: int = 10
) -> str:
    """
    Create an enhanced terminal session with template and collaboration support.
    
    Args:
        session_id: Unique session identifier
        session_type: Session type (standard, template, collaborative)
        working_directory: Initial working directory
        environment_vars: Environment variables
        template_id: Template ID to apply
        agent_id: Agent creating the session
        agent_name: Agent name
        description: Session description
        max_participants: Maximum participants for collaborative sessions
    
    Returns:
        JSON string with creation result
    """
    try:
        session_type_enum = SessionType(session_type.lower())
        
        result = _enhanced_session_manager.create_session(
            session_id=session_id,
            session_type=session_type_enum,
            working_directory=working_directory,
            environment_vars=environment_vars,
            template_id=template_id,
            agent_id=agent_id,
            agent_name=agent_name,
            description=description,
            max_participants=max_participants
        )
        
        return json.dumps(result)
        
    except ValueError as e:
        return json.dumps({
            "status": "failure",
            "message": f"Invalid session type: {session_type}"
        })
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to create session: {str(e)}"
        })


@tool(args_schema=ExecuteCommandInput)
def terminal_session_execute_enhanced(
    session_id: str,
    command: str,
    agent_id: str = "system",
    timeout: int = 30,
    auto_advance_template: bool = True
) -> str:
    """
    Execute a command in an enhanced terminal session with collaboration support.
    
    Args:
        session_id: Session identifier
        command: Command to execute
        agent_id: Agent executing the command
        timeout: Command timeout in seconds
        auto_advance_template: Auto-advance template progress
    
    Returns:
        JSON string with execution result
    """
    try:
        result = _enhanced_session_manager.execute_command(
            session_id=session_id,
            command=command,
            agent_id=agent_id,
            timeout=timeout,
            auto_advance_template=auto_advance_template
        )
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to execute command: {str(e)}"
        })


@tool(args_schema=ExecuteTemplateInput)
def terminal_session_execute_template(
    session_id: str,
    template_id: str,
    agent_id: str = "system",
    timeout: int = 30,
    interactive: bool = False
) -> str:
    """
    Execute a session template with pre-configured commands.
    
    Args:
        session_id: Session identifier
        template_id: Template ID to execute
        agent_id: Agent executing the template
        timeout: Command timeout in seconds
        interactive: Interactive template execution
    
    Returns:
        JSON string with execution result
    """
    try:
        result = _enhanced_session_manager.execute_template(
            session_id=session_id,
            template_id=template_id,
            agent_id=agent_id,
            timeout=timeout,
            interactive=interactive
        )
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to execute template: {str(e)}"
        })


@tool(args_schema=JoinSessionInput)
def terminal_session_join(
    session_id: str,
    agent_id: str,
    agent_name: str,
    role: str = "collaborator"
) -> str:
    """
    Join a collaborative terminal session.
    
    Args:
        session_id: Session identifier
        agent_id: Agent joining the session
        agent_name: Agent name
        role: Agent role (owner, collaborator, observer, admin)
    
    Returns:
        JSON string with join result
    """
    try:
        role_enum = AgentRole(role.lower())
        
        result = _enhanced_session_manager.join_session(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            role=role_enum
        )
        
        return json.dumps(result)
        
    except ValueError as e:
        return json.dumps({
            "status": "failure",
            "message": f"Invalid role: {role}"
        })
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to join session: {str(e)}"
        })


@tool(args_schema=SessionLockInput)
def terminal_session_lock(session_id: str, agent_id: str) -> str:
    """
    Lock a terminal session for exclusive access.
    
    Args:
        session_id: Session identifier
        agent_id: Agent requesting the lock
    
    Returns:
        JSON string with lock result
    """
    try:
        result = _enhanced_session_manager.lock_session(session_id, agent_id)
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to lock session: {str(e)}"
        })


@tool(args_schema=SessionLockInput)
def terminal_session_unlock(session_id: str, agent_id: str) -> str:
    """
    Unlock a terminal session.
    
    Args:
        session_id: Session identifier
        agent_id: Agent requesting the unlock
    
    Returns:
        JSON string with unlock result
    """
    try:
        result = _enhanced_session_manager.unlock_session(session_id, agent_id)
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to unlock session: {str(e)}"
        })


@tool(args_schema=CreateTemplateInput)
def terminal_session_create_template(
    name: str,
    description: str,
    category: str,
    commands: List[str],
    environment_vars: Optional[Dict[str, str]] = None,
    working_directory: Optional[str] = None,
    prerequisites: Optional[List[str]] = None,
    expected_duration: int = 30,
    created_by: str = "user",
    tags: Optional[List[str]] = None
) -> str:
    """
    Create a new session template for common workflows.
    
    Args:
        name: Template name
        description: Template description
        category: Template category
        commands: List of commands
        environment_vars: Environment variables
        working_directory: Working directory
        prerequisites: Prerequisites
        expected_duration: Expected duration in minutes
        created_by: Creator identifier
        tags: Template tags
    
    Returns:
        JSON string with creation result
    """
    try:
        category_enum = TemplateCategory(category.lower())
        
        template_id = _enhanced_session_manager.template_manager.create_template(
            name=name,
            description=description,
            category=category_enum,
            commands=commands,
            environment_vars=environment_vars,
            working_directory=working_directory,
            prerequisites=prerequisites,
            expected_duration=expected_duration,
            created_by=created_by,
            tags=tags
        )
        
        return json.dumps({
            "status": "success",
            "data": {"template_id": template_id},
            "message": f"Template '{name}' created successfully"
        })
        
    except ValueError as e:
        return json.dumps({
            "status": "failure",
            "message": f"Invalid category: {category}"
        })
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to create template: {str(e)}"
        })


@tool(args_schema=ListTemplatesInput)
def terminal_session_list_templates(
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    List available session templates with optional filtering.
    
    Args:
        category: Filter by category
        tags: Filter by tags
    
    Returns:
        JSON string with templates list
    """
    try:
        category_enum = None
        if category:
            category_enum = TemplateCategory(category.lower())
        
        templates = _enhanced_session_manager.template_manager.list_templates(
            category=category_enum,
            tags=tags
        )
        
        return json.dumps({
            "status": "success",
            "data": {
                "templates": [t.to_dict() for t in templates],
                "count": len(templates)
            },
            "message": f"Found {len(templates)} templates"
        })
        
    except ValueError as e:
        return json.dumps({
            "status": "failure",
            "message": f"Invalid category: {category}"
        })
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to list templates: {str(e)}"
        })


@tool
def terminal_session_get_collaboration_events(
    session_id: str,
    agent_id: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    Get collaboration events for a session.
    
    Args:
        session_id: Session identifier
        agent_id: Filter by agent ID
        limit: Maximum number of events to return
    
    Returns:
        JSON string with collaboration events
    """
    try:
        result = _enhanced_session_manager.get_collaboration_events(
            session_id=session_id,
            agent_id=agent_id,
            limit=limit
        )
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get collaboration events: {str(e)}"
        })


@tool
def terminal_session_get_info_enhanced(session_id: str) -> str:
    """
    Get detailed information about an enhanced terminal session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        JSON string with session information
    """
    try:
        if session_id not in _enhanced_session_manager.sessions:
            return json.dumps({
                "status": "failure",
                "message": f"Session '{session_id}' not found"
            })
        
        session = _enhanced_session_manager.sessions[session_id]
        
        # Get template info if applicable
        template_info = None
        if session.template_id:
            template = _enhanced_session_manager.template_manager.get_template(session.template_id)
            if template:
                template_info = {
                    "template_id": template.template_id,
                    "name": template.name,
                    "progress": f"{session.template_progress}/{len(template.commands)}",
                    "completed": session.template_progress >= len(template.commands)
                }
        
        # Get participant info
        participants_info = []
        for participant in session.participants.values():
            if participant.active:
                participants_info.append({
                    "agent_id": participant.agent_id,
                    "agent_name": participant.agent_name,
                    "role": participant.role.value,
                    "joined_at": participant.joined_at.isoformat(),
                    "last_activity": participant.last_activity.isoformat()
                })
        
        return json.dumps({
            "status": "success",
            "data": {
                "session_id": session_id,
                "session_type": session.session_type.value,
                "working_directory": session.working_directory,
                "environment_vars": session.environment_vars,
                "command_history_count": len(session.command_history),
                "created_at": session.created_at.isoformat(),
                "last_used": session.last_used.isoformat(),
                "description": session.description,
                "tags": session.tags,
                "template_info": template_info,
                "participants": participants_info,
                "locked_by": session.locked_by,
                "locked_at": session.locked_at.isoformat() if session.locked_at else None,
                "collaboration_events_count": len(session.collaboration_events)
            },
            "message": f"Enhanced session '{session_id}' information retrieved"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get session info: {str(e)}"
        })


# Export enhanced session manager for direct use
__all__ = [
    "EnhancedTerminalSessionManager",
    "SessionTemplateManager", 
    "terminal_session_create_enhanced",
    "terminal_session_execute_enhanced",
    "terminal_session_execute_template",
    "terminal_session_join",
    "terminal_session_lock",
    "terminal_session_unlock",
    "terminal_session_create_template",
    "terminal_session_list_templates",
    "terminal_session_get_collaboration_events",
    "terminal_session_get_info_enhanced"
]