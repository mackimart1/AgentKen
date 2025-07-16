"""
Enhanced Scratchpad Tool with Persistence, Access Control, and Versioning

Key Enhancements:
1. Persistence: Data persists across sessions with disk/database storage
2. Access Control: Fine-grained permissions for agent-specific access
3. Versioning: Complete history tracking with rollback capabilities

This enhanced version provides enterprise-grade data management for AI agents.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for scratchpad access."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class StorageBackend(Enum):
    """Storage backend options."""
    MEMORY = "memory"
    DISK = "disk"
    DATABASE = "database"


@dataclass
class ScratchpadEntry:
    """Represents a scratchpad entry with metadata."""
    key: str
    value: str
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    permissions: Dict[str, List[str]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "permissions": self.permissions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchpadEntry':
        return cls(
            key=data["key"],
            value=data["value"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            created_by=data["created_by"],
            updated_by=data["updated_by"],
            permissions=data["permissions"],
            metadata=data["metadata"]
        )


@dataclass
class VersionHistory:
    """Represents version history for a key."""
    key: str
    version: int
    value: str
    timestamp: datetime
    agent: str
    action: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "action": self.action,
            "metadata": self.metadata
        }


class EnhancedScratchpad:
    """Enhanced scratchpad with persistence, access control, and versioning."""
    
    def __init__(self, 
                 storage_backend: StorageBackend = StorageBackend.DATABASE,
                 storage_path: str = "scratchpad_data",
                 enable_encryption: bool = False):
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path)
        self.enable_encryption = enable_encryption
        self._lock = threading.RLock()
        
        # In-memory storage for fast access
        self._memory_storage: Dict[str, ScratchpadEntry] = {}
        self._version_history: Dict[str, List[VersionHistory]] = {}
        
        # Initialize storage backend
        self._init_storage()
        self._load_data()
    
    def _init_storage(self):
        """Initialize the storage backend."""
        if self.storage_backend == StorageBackend.DISK:
            self.storage_path.mkdir(exist_ok=True)
            self._data_file = self.storage_path / "scratchpad_data.json"
            self._history_file = self.storage_path / "scratchpad_history.json"
        
        elif self.storage_backend == StorageBackend.DATABASE:
            self.storage_path.mkdir(exist_ok=True)
            self._db_file = self.storage_path / "scratchpad.db"
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self._db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scratchpad_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_key_version 
                ON version_history(key, version)
            """)
    
    def _load_data(self):
        """Load data from storage backend."""
        try:
            if self.storage_backend == StorageBackend.DISK:
                self._load_from_disk()
            elif self.storage_backend == StorageBackend.DATABASE:
                self._load_from_database()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def _load_from_disk(self):
        """Load data from disk files."""
        # Load main data
        if self._data_file.exists():
            with open(self._data_file, 'r') as f:
                data = json.load(f)
                for key, entry_data in data.items():
                    self._memory_storage[key] = ScratchpadEntry.from_dict(entry_data)
        
        # Load history
        if self._history_file.exists():
            with open(self._history_file, 'r') as f:
                history_data = json.load(f)
                for key, versions in history_data.items():
                    self._version_history[key] = [
                        VersionHistory(
                            key=v["key"],
                            version=v["version"],
                            value=v["value"],
                            timestamp=datetime.fromisoformat(v["timestamp"]),
                            agent=v["agent"],
                            action=v["action"],
                            metadata=v["metadata"]
                        ) for v in versions
                    ]
    
    def _load_from_database(self):
        """Load data from SQLite database."""
        with sqlite3.connect(self._db_file) as conn:
            # Load main data
            cursor = conn.execute("SELECT * FROM scratchpad_entries")
            for row in cursor.fetchall():
                key, value, version, created_at, updated_at, created_by, updated_by, permissions, metadata = row
                entry = ScratchpadEntry(
                    key=key,
                    value=value,
                    version=version,
                    created_at=datetime.fromisoformat(created_at),
                    updated_at=datetime.fromisoformat(updated_at),
                    created_by=created_by,
                    updated_by=updated_by,
                    permissions=json.loads(permissions),
                    metadata=json.loads(metadata)
                )
                self._memory_storage[key] = entry
            
            # Load history
            cursor = conn.execute("SELECT * FROM version_history ORDER BY key, version")
            for row in cursor.fetchall():
                _, key, version, value, timestamp, agent, action, metadata = row
                history = VersionHistory(
                    key=key,
                    version=version,
                    value=value,
                    timestamp=datetime.fromisoformat(timestamp),
                    agent=agent,
                    action=action,
                    metadata=json.loads(metadata)
                )
                
                if key not in self._version_history:
                    self._version_history[key] = []
                self._version_history[key].append(history)
    
    def _save_data(self):
        """Save data to storage backend."""
        try:
            if self.storage_backend == StorageBackend.DISK:
                self._save_to_disk()
            elif self.storage_backend == StorageBackend.DATABASE:
                self._save_to_database()
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def _save_to_disk(self):
        """Save data to disk files."""
        # Save main data
        data = {key: entry.to_dict() for key, entry in self._memory_storage.items()}
        with open(self._data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save history
        history_data = {
            key: [v.to_dict() for v in versions]
            for key, versions in self._version_history.items()
        }
        with open(self._history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _save_to_database(self):
        """Save data to SQLite database."""
        with sqlite3.connect(self._db_file) as conn:
            # This is handled incrementally in other methods
            pass
    
    def _save_entry_to_db(self, entry: ScratchpadEntry):
        """Save a single entry to database."""
        if self.storage_backend != StorageBackend.DATABASE:
            return
        
        with sqlite3.connect(self._db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scratchpad_entries 
                (key, value, version, created_at, updated_at, created_by, updated_by, permissions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.key,
                entry.value,
                entry.version,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.created_by,
                entry.updated_by,
                json.dumps(entry.permissions),
                json.dumps(entry.metadata)
            ))
    
    def _save_history_to_db(self, history: VersionHistory):
        """Save version history to database."""
        if self.storage_backend != StorageBackend.DATABASE:
            return
        
        with sqlite3.connect(self._db_file) as conn:
            conn.execute("""
                INSERT INTO version_history 
                (key, version, value, timestamp, agent, action, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                history.key,
                history.version,
                history.value,
                history.timestamp.isoformat(),
                history.agent,
                history.action,
                json.dumps(history.metadata)
            ))
    
    def _check_permission(self, key: str, agent: str, permission: PermissionLevel) -> bool:
        """Check if agent has permission for the key."""
        if key not in self._memory_storage:
            return True  # Allow creation
        
        entry = self._memory_storage[key]
        
        # Creator and admin always have full access
        if agent == entry.created_by or agent == "admin":
            return True
        
        # Check specific permissions
        agent_permissions = entry.permissions.get(agent, [])
        
        # Admin permission grants all access
        if PermissionLevel.ADMIN.value in agent_permissions:
            return True
        
        # Check specific permission
        if permission.value in agent_permissions:
            return True
        
        # Write permission includes read
        if permission == PermissionLevel.READ and PermissionLevel.WRITE.value in agent_permissions:
            return True
        
        return False
    
    def _add_version_history(self, key: str, value: str, agent: str, action: str, metadata: Dict[str, Any] = None):
        """Add entry to version history."""
        if key not in self._version_history:
            self._version_history[key] = []
        
        version = len(self._version_history[key]) + 1
        history = VersionHistory(
            key=key,
            version=version,
            value=value,
            timestamp=datetime.now(),
            agent=agent,
            action=action,
            metadata=metadata or {}
        )
        
        self._version_history[key].append(history)
        self._save_history_to_db(history)
        
        return version
    
    def write(self, key: str, value: str, agent: str = "system", 
              permissions: Dict[str, List[str]] = None, 
              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Write a value to the scratchpad."""
        with self._lock:
            # Check permissions
            if not self._check_permission(key, agent, PermissionLevel.WRITE):
                return {
                    "status": "error",
                    "message": f"Agent '{agent}' does not have write permission for key '{key}'"
                }
            
            now = datetime.now()
            
            if key in self._memory_storage:
                # Update existing entry
                entry = self._memory_storage[key]
                old_value = entry.value
                entry.value = value
                entry.version += 1
                entry.updated_at = now
                entry.updated_by = agent
                
                if metadata:
                    entry.metadata.update(metadata)
                
                action = "update"
            else:
                # Create new entry
                entry = ScratchpadEntry(
                    key=key,
                    value=value,
                    version=1,
                    created_at=now,
                    updated_at=now,
                    created_by=agent,
                    updated_by=agent,
                    permissions=permissions or {agent: [PermissionLevel.ADMIN.value]},
                    metadata=metadata or {}
                )
                self._memory_storage[key] = entry
                action = "create"
            
            # Add to version history
            version = self._add_version_history(key, value, agent, action, metadata)
            
            # Save to persistent storage
            self._save_entry_to_db(entry)
            if self.storage_backend == StorageBackend.DISK:
                self._save_data()
            
            return {
                "status": "success",
                "key": key,
                "value": value,
                "version": entry.version,
                "action": action,
                "timestamp": now.isoformat()
            }
    
    def read(self, key: str, agent: str = "system", version: Optional[int] = None) -> Dict[str, Any]:
        """Read a value from the scratchpad."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check permissions
            if not self._check_permission(key, agent, PermissionLevel.READ):
                return {
                    "status": "error",
                    "message": f"Agent '{agent}' does not have read permission for key '{key}'"
                }
            
            entry = self._memory_storage[key]
            
            if version is None:
                # Return current version
                return {
                    "status": "success",
                    "key": key,
                    "value": entry.value,
                    "version": entry.version,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "created_by": entry.created_by,
                    "updated_by": entry.updated_by,
                    "metadata": entry.metadata
                }
            else:
                # Return specific version
                if key not in self._version_history:
                    return {
                        "status": "error",
                        "message": f"No version history found for key '{key}'"
                    }
                
                history_entries = self._version_history[key]
                if version < 1 or version > len(history_entries):
                    return {
                        "status": "error",
                        "message": f"Version {version} not found for key '{key}'"
                    }
                
                history = history_entries[version - 1]
                return {
                    "status": "success",
                    "key": key,
                    "value": history.value,
                    "version": history.version,
                    "timestamp": history.timestamp.isoformat(),
                    "agent": history.agent,
                    "action": history.action,
                    "metadata": history.metadata
                }
    
    def delete(self, key: str, agent: str = "system") -> Dict[str, Any]:
        """Delete a key from the scratchpad."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check permissions
            if not self._check_permission(key, agent, PermissionLevel.DELETE):
                return {
                    "status": "error",
                    "message": f"Agent '{agent}' does not have delete permission for key '{key}'"
                }
            
            entry = self._memory_storage[key]
            
            # Add to version history
            self._add_version_history(key, entry.value, agent, "delete")
            
            # Remove from memory
            del self._memory_storage[key]
            
            # Remove from database
            if self.storage_backend == StorageBackend.DATABASE:
                with sqlite3.connect(self._db_file) as conn:
                    conn.execute("DELETE FROM scratchpad_entries WHERE key = ?", (key,))
            elif self.storage_backend == StorageBackend.DISK:
                self._save_data()
            
            return {
                "status": "success",
                "message": f"Key '{key}' deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
    
    def list_keys(self, agent: str = "system", include_metadata: bool = False) -> Dict[str, Any]:
        """List all keys the agent has access to."""
        with self._lock:
            accessible_keys = []
            
            for key, entry in self._memory_storage.items():
                if self._check_permission(key, agent, PermissionLevel.READ):
                    if include_metadata:
                        accessible_keys.append({
                            "key": key,
                            "version": entry.version,
                            "created_at": entry.created_at.isoformat(),
                            "updated_at": entry.updated_at.isoformat(),
                            "created_by": entry.created_by,
                            "updated_by": entry.updated_by,
                            "metadata": entry.metadata
                        })
                    else:
                        accessible_keys.append(key)
            
            return {
                "status": "success",
                "keys": accessible_keys,
                "count": len(accessible_keys)
            }
    
    def get_history(self, key: str, agent: str = "system", limit: Optional[int] = None) -> Dict[str, Any]:
        """Get version history for a key."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check permissions
            if not self._check_permission(key, agent, PermissionLevel.READ):
                return {
                    "status": "error",
                    "message": f"Agent '{agent}' does not have read permission for key '{key}'"
                }
            
            if key not in self._version_history:
                return {
                    "status": "success",
                    "key": key,
                    "history": [],
                    "count": 0
                }
            
            history = self._version_history[key]
            if limit:
                history = history[-limit:]  # Get most recent entries
            
            return {
                "status": "success",
                "key": key,
                "history": [h.to_dict() for h in history],
                "count": len(history)
            }
    
    def rollback(self, key: str, version: int, agent: str = "system") -> Dict[str, Any]:
        """Rollback a key to a specific version."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check permissions
            if not self._check_permission(key, agent, PermissionLevel.WRITE):
                return {
                    "status": "error",
                    "message": f"Agent '{agent}' does not have write permission for key '{key}'"
                }
            
            if key not in self._version_history:
                return {
                    "status": "error",
                    "message": f"No version history found for key '{key}'"
                }
            
            history_entries = self._version_history[key]
            if version < 1 or version > len(history_entries):
                return {
                    "status": "error",
                    "message": f"Version {version} not found for key '{key}'"
                }
            
            # Get the target version
            target_history = history_entries[version - 1]
            
            # Update current entry
            entry = self._memory_storage[key]
            old_value = entry.value
            entry.value = target_history.value
            entry.version += 1
            entry.updated_at = datetime.now()
            entry.updated_by = agent
            
            # Add rollback to history
            self._add_version_history(
                key, 
                target_history.value, 
                agent, 
                "rollback", 
                {"rollback_to_version": version, "previous_value": old_value}
            )
            
            # Save to persistent storage
            self._save_entry_to_db(entry)
            if self.storage_backend == StorageBackend.DISK:
                self._save_data()
            
            return {
                "status": "success",
                "key": key,
                "rolled_back_to_version": version,
                "new_version": entry.version,
                "value": entry.value,
                "timestamp": entry.updated_at.isoformat()
            }
    
    def set_permissions(self, key: str, agent: str, permissions: List[str], requesting_agent: str = "admin") -> Dict[str, Any]:
        """Set permissions for an agent on a specific key."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check if requesting agent has admin permission
            if not self._check_permission(key, requesting_agent, PermissionLevel.ADMIN):
                return {
                    "status": "error",
                    "message": f"Agent '{requesting_agent}' does not have admin permission for key '{key}'"
                }
            
            # Validate permissions
            valid_permissions = [p.value for p in PermissionLevel]
            invalid_perms = [p for p in permissions if p not in valid_permissions]
            if invalid_perms:
                return {
                    "status": "error",
                    "message": f"Invalid permissions: {invalid_perms}. Valid: {valid_permissions}"
                }
            
            entry = self._memory_storage[key]
            entry.permissions[agent] = permissions
            entry.updated_at = datetime.now()
            entry.updated_by = requesting_agent
            
            # Add to history
            self._add_version_history(
                key,
                entry.value,
                requesting_agent,
                "permission_change",
                {"target_agent": agent, "new_permissions": permissions}
            )
            
            # Save to persistent storage
            self._save_entry_to_db(entry)
            if self.storage_backend == StorageBackend.DISK:
                self._save_data()
            
            return {
                "status": "success",
                "key": key,
                "agent": agent,
                "permissions": permissions,
                "timestamp": entry.updated_at.isoformat()
            }
    
    def get_permissions(self, key: str, requesting_agent: str = "admin") -> Dict[str, Any]:
        """Get permissions for a key."""
        with self._lock:
            if key not in self._memory_storage:
                return {
                    "status": "error",
                    "message": f"Key '{key}' not found"
                }
            
            # Check if requesting agent has read permission
            if not self._check_permission(key, requesting_agent, PermissionLevel.READ):
                return {
                    "status": "error",
                    "message": f"Agent '{requesting_agent}' does not have read permission for key '{key}'"
                }
            
            entry = self._memory_storage[key]
            
            return {
                "status": "success",
                "key": key,
                "permissions": entry.permissions,
                "created_by": entry.created_by
            }
    
    def clear_all(self, agent: str = "admin") -> Dict[str, Any]:
        """Clear all data (admin only)."""
        with self._lock:
            if agent != "admin":
                return {
                    "status": "error",
                    "message": "Only admin can clear all data"
                }
            
            # Clear memory
            cleared_keys = list(self._memory_storage.keys())
            self._memory_storage.clear()
            self._version_history.clear()
            
            # Clear persistent storage
            if self.storage_backend == StorageBackend.DATABASE:
                with sqlite3.connect(self._db_file) as conn:
                    conn.execute("DELETE FROM scratchpad_entries")
                    conn.execute("DELETE FROM version_history")
            elif self.storage_backend == StorageBackend.DISK:
                if self._data_file.exists():
                    self._data_file.unlink()
                if self._history_file.exists():
                    self._history_file.unlink()
            
            return {
                "status": "success",
                "message": f"Cleared {len(cleared_keys)} keys",
                "cleared_keys": cleared_keys,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scratchpad statistics."""
        with self._lock:
            total_keys = len(self._memory_storage)
            total_versions = sum(len(history) for history in self._version_history.values())
            
            # Calculate storage size
            if self.storage_backend == StorageBackend.DATABASE and self._db_file.exists():
                storage_size = self._db_file.stat().st_size
            elif self.storage_backend == StorageBackend.DISK:
                storage_size = 0
                if self._data_file.exists():
                    storage_size += self._data_file.stat().st_size
                if self._history_file.exists():
                    storage_size += self._history_file.stat().st_size
            else:
                storage_size = 0
            
            return {
                "status": "success",
                "statistics": {
                    "total_keys": total_keys,
                    "total_versions": total_versions,
                    "storage_backend": self.storage_backend.value,
                    "storage_size_bytes": storage_size,
                    "encryption_enabled": self.enable_encryption
                }
            }


# Global enhanced scratchpad instance
_enhanced_scratchpad = EnhancedScratchpad()


# Input models for the enhanced tools
class WriteInput(BaseModel):
    key: str = Field(description="Key to write to")
    value: str = Field(description="Value to store")
    agent: str = Field(default="system", description="Agent performing the operation")
    permissions: Optional[Dict[str, List[str]]] = Field(default=None, description="Permissions for the key")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ReadInput(BaseModel):
    key: str = Field(description="Key to read from")
    agent: str = Field(default="system", description="Agent performing the operation")
    version: Optional[int] = Field(default=None, description="Specific version to read")


class DeleteInput(BaseModel):
    key: str = Field(description="Key to delete")
    agent: str = Field(default="system", description="Agent performing the operation")


class ListKeysInput(BaseModel):
    agent: str = Field(default="system", description="Agent performing the operation")
    include_metadata: bool = Field(default=False, description="Include metadata in response")


class GetHistoryInput(BaseModel):
    key: str = Field(description="Key to get history for")
    agent: str = Field(default="system", description="Agent performing the operation")
    limit: Optional[int] = Field(default=None, description="Limit number of history entries")


class RollbackInput(BaseModel):
    key: str = Field(description="Key to rollback")
    version: int = Field(description="Version to rollback to")
    agent: str = Field(default="system", description="Agent performing the operation")


class SetPermissionsInput(BaseModel):
    key: str = Field(description="Key to set permissions for")
    agent: str = Field(description="Agent to set permissions for")
    permissions: List[str] = Field(description="List of permissions to grant")
    requesting_agent: str = Field(default="admin", description="Agent requesting the permission change")


class GetPermissionsInput(BaseModel):
    key: str = Field(description="Key to get permissions for")
    requesting_agent: str = Field(default="admin", description="Agent requesting the permissions")


class ClearAllInput(BaseModel):
    agent: str = Field(default="admin", description="Agent performing the operation")


# Enhanced scratchpad tools
@tool(args_schema=WriteInput)
def scratchpad_write(
    key: str, 
    value: str, 
    agent: str = "system", 
    permissions: Optional[Dict[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Write a value to the enhanced scratchpad with persistence and access control.
    
    Args:
        key: Key to write to
        value: Value to store
        agent: Agent performing the operation
        permissions: Permissions for the key (for new keys)
        metadata: Additional metadata
    
    Returns:
        JSON string with operation result
    """
    try:
        result = _enhanced_scratchpad.write(key, value, agent, permissions, metadata)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Write operation failed: {str(e)}"
        })


@tool(args_schema=ReadInput)
def scratchpad_read(key: str, agent: str = "system", version: Optional[int] = None) -> str:
    """
    Read a value from the enhanced scratchpad with version support.
    
    Args:
        key: Key to read from
        agent: Agent performing the operation
        version: Specific version to read (None for latest)
    
    Returns:
        JSON string with the value and metadata
    """
    try:
        result = _enhanced_scratchpad.read(key, agent, version)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Read operation failed: {str(e)}"
        })


@tool(args_schema=DeleteInput)
def scratchpad_delete(key: str, agent: str = "system") -> str:
    """
    Delete a key from the enhanced scratchpad with access control.
    
    Args:
        key: Key to delete
        agent: Agent performing the operation
    
    Returns:
        JSON string with operation result
    """
    try:
        result = _enhanced_scratchpad.delete(key, agent)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Delete operation failed: {str(e)}"
        })


@tool(args_schema=ListKeysInput)
def scratchpad_list_keys(agent: str = "system", include_metadata: bool = False) -> str:
    """
    List all keys accessible to the agent.
    
    Args:
        agent: Agent performing the operation
        include_metadata: Include metadata in response
    
    Returns:
        JSON string with accessible keys
    """
    try:
        result = _enhanced_scratchpad.list_keys(agent, include_metadata)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"List keys operation failed: {str(e)}"
        })


@tool(args_schema=GetHistoryInput)
def scratchpad_get_history(key: str, agent: str = "system", limit: Optional[int] = None) -> str:
    """
    Get version history for a key.
    
    Args:
        key: Key to get history for
        agent: Agent performing the operation
        limit: Limit number of history entries
    
    Returns:
        JSON string with version history
    """
    try:
        result = _enhanced_scratchpad.get_history(key, agent, limit)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Get history operation failed: {str(e)}"
        })


@tool(args_schema=RollbackInput)
def scratchpad_rollback(key: str, version: int, agent: str = "system") -> str:
    """
    Rollback a key to a specific version.
    
    Args:
        key: Key to rollback
        version: Version to rollback to
        agent: Agent performing the operation
    
    Returns:
        JSON string with rollback result
    """
    try:
        result = _enhanced_scratchpad.rollback(key, version, agent)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Rollback operation failed: {str(e)}"
        })


@tool(args_schema=SetPermissionsInput)
def scratchpad_set_permissions(
    key: str, 
    agent: str, 
    permissions: List[str], 
    requesting_agent: str = "admin"
) -> str:
    """
    Set permissions for an agent on a specific key.
    
    Args:
        key: Key to set permissions for
        agent: Agent to set permissions for
        permissions: List of permissions to grant (read, write, delete, admin)
        requesting_agent: Agent requesting the permission change
    
    Returns:
        JSON string with operation result
    """
    try:
        result = _enhanced_scratchpad.set_permissions(key, agent, permissions, requesting_agent)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Set permissions operation failed: {str(e)}"
        })


@tool(args_schema=GetPermissionsInput)
def scratchpad_get_permissions(key: str, requesting_agent: str = "admin") -> str:
    """
    Get permissions for a key.
    
    Args:
        key: Key to get permissions for
        requesting_agent: Agent requesting the permissions
    
    Returns:
        JSON string with permissions information
    """
    try:
        result = _enhanced_scratchpad.get_permissions(key, requesting_agent)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Get permissions operation failed: {str(e)}"
        })


@tool(args_schema=ClearAllInput)
def scratchpad_clear_all(agent: str = "admin") -> str:
    """
    Clear all scratchpad data (admin only).
    
    Args:
        agent: Agent performing the operation
    
    Returns:
        JSON string with operation result
    """
    try:
        result = _enhanced_scratchpad.clear_all(agent)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Clear all operation failed: {str(e)}"
        })


@tool
def scratchpad_get_stats() -> str:
    """
    Get scratchpad statistics and information.
    
    Returns:
        JSON string with statistics
    """
    try:
        result = _enhanced_scratchpad.get_stats()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Get stats operation failed: {str(e)}"
        })


# Export enhanced scratchpad instance for direct use
__all__ = [
    "EnhancedScratchpad", 
    "scratchpad_write", 
    "scratchpad_read", 
    "scratchpad_delete",
    "scratchpad_list_keys", 
    "scratchpad_get_history", 
    "scratchpad_rollback",
    "scratchpad_set_permissions", 
    "scratchpad_get_permissions", 
    "scratchpad_clear_all",
    "scratchpad_get_stats"
]