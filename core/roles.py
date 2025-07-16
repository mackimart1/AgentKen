"""
Role and Permission Management System

Defines user roles, permissions, and decorators for enforcing access control
in the agent/tool creation system.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
from enum import Enum
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Available user roles in the system"""

    TOOL_MAKER = "tool_maker"
    AGENT_SMITH = "agent_smith"
    ADMIN = "admin"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """Available permissions in the system"""

    CREATE_AGENT = "create_agent"
    CREATE_TOOL = "create_tool"
    MODIFY_AGENT = "modify_agent"
    MODIFY_TOOL = "modify_tool"
    DELETE_AGENT = "delete_agent"
    DELETE_TOOL = "delete_tool"
    VIEW_AGENTS = "view_agents"
    VIEW_TOOLS = "view_tools"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"


class RolePermissionManager:
    """Manages role-based permissions and user access control"""

    def __init__(self, users_file: str = "users.json", roles_file: str = "roles.json"):
        self.users_file = users_file
        self.roles_file = roles_file
        self._role_permissions = self._load_role_permissions()
        self._users = self._load_users()

    def _load_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Load role-permission mappings from file or use defaults"""
        default_permissions = {
            UserRole.ADMIN: [
                Permission.CREATE_AGENT,
                Permission.CREATE_TOOL,
                Permission.MODIFY_AGENT,
                Permission.MODIFY_TOOL,
                Permission.DELETE_AGENT,
                Permission.DELETE_TOOL,
                Permission.VIEW_AGENTS,
                Permission.VIEW_TOOLS,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
            ],
            UserRole.AGENT_SMITH: [
                Permission.CREATE_AGENT,
                Permission.MODIFY_AGENT,
                Permission.VIEW_AGENTS,
                Permission.VIEW_TOOLS,
            ],
            UserRole.TOOL_MAKER: [
                Permission.CREATE_TOOL,
                Permission.MODIFY_TOOL,
                Permission.VIEW_AGENTS,
                Permission.VIEW_TOOLS,
            ],
            UserRole.VIEWER: [Permission.VIEW_AGENTS, Permission.VIEW_TOOLS],
            UserRole.GUEST: [],
        }

        try:
            if os.path.exists(self.roles_file):
                with open(self.roles_file, "r") as f:
                    data = json.load(f)
                    # Convert string keys back to UserRole enum
                    return {
                        UserRole(role_str): [Permission(perm_str) for perm_str in perms]
                        for role_str, perms in data.items()
                    }
        except Exception as e:
            logger.warning(f"Failed to load roles file, using defaults: {e}")

        return default_permissions

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load user data from file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load users file: {e}")

        return {}

    def _save_users(self):
        """Save user data to file"""
        try:
            with open(self.users_file, "w") as f:
                json.dump(self._users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users file: {e}")

    def _save_roles(self):
        """Save role permissions to file"""
        try:
            # Convert enum keys to strings for JSON serialization
            data = {
                role.value: [perm.value for perm in perms]
                for role, perms in self._role_permissions.items()
            }
            with open(self.roles_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save roles file: {e}")

    def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """Get the role for a specific user"""
        user_data = self._users.get(user_id)
        if user_data:
            role_str = user_data.get("role")
            if role_str:
                try:
                    return UserRole(role_str)
                except ValueError:
                    logger.warning(f"Invalid role '{role_str}' for user {user_id}")
        return None

    def set_user_role(self, user_id: str, role: UserRole, created_by: str = "system"):
        """Set the role for a specific user"""
        if user_id not in self._users:
            self._users[user_id] = {}

        self._users[user_id].update(
            {
                "role": role.value,
                "created_at": datetime.now().isoformat(),
                "created_by": created_by,
                "last_updated": datetime.now().isoformat(),
            }
        )

        self._save_users()
        logger.info(f"Set role {role.value} for user {user_id}")

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission"""
        role = self.get_user_role(user_id)
        if not role:
            logger.warning(f"No role found for user {user_id}")
            return False

        user_permissions = self._role_permissions.get(role, [])
        has_permission = permission in user_permissions

        logger.debug(
            f"Permission check: user={user_id}, role={role.value}, "
            f"permission={permission.value}, granted={has_permission}"
        )

        return has_permission

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for a specific user"""
        role = self.get_user_role(user_id)
        if not role:
            return []

        return self._role_permissions.get(role, [])

    def add_role_permission(self, role: UserRole, permission: Permission):
        """Add a permission to a role"""
        if role not in self._role_permissions:
            self._role_permissions[role] = []

        if permission not in self._role_permissions[role]:
            self._role_permissions[role].append(permission)
            self._save_roles()
            logger.info(f"Added permission {permission.value} to role {role.value}")

    def remove_role_permission(self, role: UserRole, permission: Permission):
        """Remove a permission from a role"""
        if (
            role in self._role_permissions
            and permission in self._role_permissions[role]
        ):
            self._role_permissions[role].remove(permission)
            self._save_roles()
            logger.info(f"Removed permission {permission.value} from role {role.value}")

    def list_users(self) -> Dict[str, Dict[str, Any]]:
        """List all users and their roles"""
        return self._users.copy()

    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        if user_id in self._users:
            del self._users[user_id]
            self._save_users()
            logger.info(f"Deleted user {user_id}")
            return True
        return False


# Global instance
role_manager = RolePermissionManager()


def requires_permission(permission: Permission):
    """Decorator to require a specific permission for function execution"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, user_id: str = None, **kwargs):
            if not user_id:
                raise ValueError("user_id parameter is required for permission checks")

            if not role_manager.check_permission(user_id, permission):
                raise PermissionError(
                    f"User {user_id} lacks required permission: {permission.value}"
                )

            return func(*args, user_id=user_id, **kwargs)

        return wrapper

    return decorator


def requires_role(role: UserRole):
    """Decorator to require a specific role for function execution"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, user_id: str = None, **kwargs):
            if not user_id:
                raise ValueError("user_id parameter is required for role checks")

            user_role = role_manager.get_user_role(user_id)
            if user_role != role:
                raise PermissionError(
                    f"User {user_id} has role {user_role.value if user_role else 'none'}, "
                    f"but {role.value} is required"
                )

            return func(*args, user_id=user_id, **kwargs)

        return wrapper

    return decorator


def log_audit_event(
    user_id: str, action: str, resource: str, details: Dict[str, Any] = None
):
    """Log an audit event for tracking user actions"""
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "details": details or {},
        "user_role": (
            role_manager.get_user_role(user_id).value
            if role_manager.get_user_role(user_id)
            else None
        ),
    }

    logger.info(f"AUDIT: {audit_entry}")

    # Could also write to a dedicated audit log file
    audit_file = "audit.log"
    try:
        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")


def create_default_admin():
    """Create a default admin user if no users exist"""
    if not role_manager.list_users():
        admin_id = "admin"
        role_manager.set_user_role(admin_id, UserRole.ADMIN, created_by="system")
        logger.info(f"Created default admin user: {admin_id}")
        return admin_id
    return None


# Initialize default admin if needed
create_default_admin()
