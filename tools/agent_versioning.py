from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path


class CreateVersionInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to version")
    file_path: str = Field(description="Path to the agent file")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the version")
    description: str = Field(default="", description="Description of changes in this version")


class RollbackVersionInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to rollback")
    version: str = Field(description="Version to rollback to (e.g., 'v1.0')")


class ListVersionsInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to list versions for")


class CompareVersionsInput(BaseModel):
    agent_name: str = Field(description="Name of the agent")
    version1: str = Field(description="First version to compare")
    version2: str = Field(description="Second version to compare")


# Global version storage
_version_registry: Dict[str, List[Dict[str, Any]]] = {}
_versions_base_path = Path("agents/.versions")
_versions_base_path.mkdir(exist_ok=True)


def _load_version_registry() -> Dict[str, List[Dict[str, Any]]]:
    """Load version registry from disk."""
    registry_file = _versions_base_path / "registry.json"
    if registry_file.exists():
        try:
            with open(registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load version registry: {e}")
    return {}


def _save_version_registry():
    """Save version registry to disk."""
    registry_file = _versions_base_path / "registry.json"
    try:
        with open(registry_file, 'w') as f:
            json.dump(_version_registry, f, indent=2)
    except Exception as e:
        print(f"Failed to save version registry: {e}")


def _calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"Failed to calculate hash for {file_path}: {e}")
    return ""


def _generate_version_number(agent_name: str) -> str:
    """Generate next version number for an agent."""
    if agent_name not in _version_registry:
        return "v1.0"
    
    versions = _version_registry[agent_name]
    if not versions:
        return "v1.0"
    
    # Extract version numbers and find the highest
    version_numbers = []
    for version_info in versions:
        version_str = version_info.get("version", "v1.0")
        try:
            # Extract major.minor from "vX.Y" format
            version_part = version_str[1:]  # Remove 'v'
            major, minor = map(int, version_part.split('.'))
            version_numbers.append((major, minor))
        except:
            version_numbers.append((1, 0))
    
    if version_numbers:
        max_major, max_minor = max(version_numbers)
        return f"v{max_major}.{max_minor + 1}"
    else:
        return "v1.0"


# Load existing registry
_version_registry = _load_version_registry()


@tool(args_schema=CreateVersionInput)
def create_agent_version(
    agent_name: str,
    file_path: str,
    metadata: Dict[str, Any] = None,
    description: str = ""
) -> str:
    """
    Create a new version of an agent with version control.
    
    Args:
        agent_name: Name of the agent to version
        file_path: Path to the agent file
        metadata: Additional metadata for the version
        description: Description of changes in this version
    
    Returns:
        JSON string with version creation result
    """
    try:
        if metadata is None:
            metadata = {}
        
        # Validate agent file exists
        if not os.path.exists(file_path):
            return json.dumps({
                "status": "failure",
                "message": f"Agent file not found: {file_path}"
            })
        
        # Initialize agent in registry if not exists
        if agent_name not in _version_registry:
            _version_registry[agent_name] = []
        
        # Generate version number
        version = _generate_version_number(agent_name)
        
        # Create version directory
        version_dir = _versions_base_path / agent_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy agent file to version directory
        version_file = version_dir / f"{agent_name}.py"
        shutil.copy2(file_path, version_file)
        
        # Calculate file hash
        file_hash = _calculate_file_hash(str(version_file))
        
        # Create version metadata
        version_info = {
            "agent_name": agent_name,
            "version": version,
            "file_path": str(version_file),
            "original_path": file_path,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metadata": metadata,
            "hash": file_hash,
            "size": os.path.getsize(version_file) if os.path.exists(version_file) else 0
        }
        
        # Add to registry
        _version_registry[agent_name].append(version_info)
        _save_version_registry()
        
        return json.dumps({
            "status": "success",
            "version": version,
            "version_path": str(version_file),
            "hash": file_hash,
            "message": f"Created version {version} for agent {agent_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to create version: {str(e)}"
        })


@tool(args_schema=RollbackVersionInput)
def rollback_agent_version(agent_name: str, version: str) -> str:
    """
    Rollback an agent to a specific version.
    
    Args:
        agent_name: Name of the agent to rollback
        version: Version to rollback to (e.g., 'v1.0')
    
    Returns:
        JSON string with rollback result
    """
    try:
        # Check if agent exists in registry
        if agent_name not in _version_registry:
            return json.dumps({
                "status": "failure",
                "message": f"No versions found for agent: {agent_name}"
            })
        
        # Find target version
        target_version = None
        for version_info in _version_registry[agent_name]:
            if version_info["version"] == version:
                target_version = version_info
                break
        
        if not target_version:
            return json.dumps({
                "status": "failure",
                "message": f"Version {version} not found for agent {agent_name}"
            })
        
        # Check if version file exists
        version_file = target_version["file_path"]
        if not os.path.exists(version_file):
            return json.dumps({
                "status": "failure",
                "message": f"Version file not found: {version_file}"
            })
        
        # Backup current version before rollback
        current_file = f"agents/{agent_name}.py"
        if os.path.exists(current_file):
            backup_file = f"agents/.backup_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            shutil.copy2(current_file, backup_file)
        
        # Copy version file back to main location
        shutil.copy2(version_file, current_file)
        
        return json.dumps({
            "status": "success",
            "version": version,
            "message": f"Successfully rolled back {agent_name} to version {version}",
            "backup_created": backup_file if 'backup_file' in locals() else None
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to rollback: {str(e)}"
        })


@tool(args_schema=ListVersionsInput)
def list_agent_versions(agent_name: str) -> str:
    """
    List all versions of an agent.
    
    Args:
        agent_name: Name of the agent to list versions for
    
    Returns:
        JSON string with version list
    """
    try:
        if agent_name not in _version_registry:
            return json.dumps({
                "status": "success",
                "versions": [],
                "message": f"No versions found for agent: {agent_name}"
            })
        
        versions = _version_registry[agent_name]
        
        # Format version information
        version_list = []
        for version_info in versions:
            version_summary = {
                "version": version_info["version"],
                "created_at": version_info["created_at"],
                "description": version_info.get("description", ""),
                "hash": version_info.get("hash", "")[:8],  # Short hash
                "size": version_info.get("size", 0)
            }
            version_list.append(version_summary)
        
        # Sort by creation date (newest first)
        version_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return json.dumps({
            "status": "success",
            "agent_name": agent_name,
            "versions": version_list,
            "total_versions": len(version_list),
            "latest_version": version_list[0]["version"] if version_list else None,
            "message": f"Found {len(version_list)} versions for {agent_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to list versions: {str(e)}"
        })


@tool(args_schema=CompareVersionsInput)
def compare_agent_versions(agent_name: str, version1: str, version2: str) -> str:
    """
    Compare two versions of an agent.
    
    Args:
        agent_name: Name of the agent
        version1: First version to compare
        version2: Second version to compare
    
    Returns:
        JSON string with comparison result
    """
    try:
        if agent_name not in _version_registry:
            return json.dumps({
                "status": "failure",
                "message": f"No versions found for agent: {agent_name}"
            })
        
        # Find both versions
        v1_info = None
        v2_info = None
        
        for version_info in _version_registry[agent_name]:
            if version_info["version"] == version1:
                v1_info = version_info
            elif version_info["version"] == version2:
                v2_info = version_info
        
        if not v1_info:
            return json.dumps({
                "status": "failure",
                "message": f"Version {version1} not found"
            })
        
        if not v2_info:
            return json.dumps({
                "status": "failure",
                "message": f"Version {version2} not found"
            })
        
        # Compare basic metadata
        comparison = {
            "version1": {
                "version": v1_info["version"],
                "created_at": v1_info["created_at"],
                "description": v1_info.get("description", ""),
                "hash": v1_info.get("hash", ""),
                "size": v1_info.get("size", 0)
            },
            "version2": {
                "version": v2_info["version"],
                "created_at": v2_info["created_at"],
                "description": v2_info.get("description", ""),
                "hash": v2_info.get("hash", ""),
                "size": v2_info.get("size", 0)
            },
            "differences": {
                "hash_different": v1_info.get("hash") != v2_info.get("hash"),
                "size_difference": v2_info.get("size", 0) - v1_info.get("size", 0),
                "time_difference": v2_info["created_at"] > v1_info["created_at"]
            }
        }
        
        # Try to compare file contents
        try:
            v1_file = v1_info["file_path"]
            v2_file = v2_info["file_path"]
            
            if os.path.exists(v1_file) and os.path.exists(v2_file):
                with open(v1_file, 'r') as f1, open(v2_file, 'r') as f2:
                    lines1 = f1.readlines()
                    lines2 = f2.readlines()
                
                # Simple line-by-line diff
                diff_lines = []
                max_lines = max(len(lines1), len(lines2))
                
                for i in range(min(max_lines, 50)):  # Limit to first 50 lines
                    line1 = lines1[i] if i < len(lines1) else ""
                    line2 = lines2[i] if i < len(lines2) else ""
                    
                    if line1.strip() != line2.strip():
                        diff_lines.append({
                            "line_number": i + 1,
                            "version1_line": line1.strip(),
                            "version2_line": line2.strip()
                        })
                
                comparison["content_diff"] = {
                    "total_differences": len(diff_lines),
                    "sample_differences": diff_lines[:10]  # Show first 10 differences
                }
        except Exception as e:
            comparison["content_diff"] = {
                "error": f"Could not compare file contents: {e}"
            }
        
        return json.dumps({
            "status": "success",
            "comparison": comparison,
            "message": f"Compared versions {version1} and {version2} of {agent_name}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to compare versions: {str(e)}"
        })


@tool
def get_version_registry_status() -> str:
    """
    Get the status of the version registry.
    
    Returns:
        JSON string with registry status
    """
    try:
        total_agents = len(_version_registry)
        total_versions = sum(len(versions) for versions in _version_registry.values())
        
        agent_summary = []
        for agent_name, versions in _version_registry.items():
            latest_version = versions[-1] if versions else None
            agent_summary.append({
                "agent_name": agent_name,
                "version_count": len(versions),
                "latest_version": latest_version["version"] if latest_version else None,
                "latest_created": latest_version["created_at"] if latest_version else None
            })
        
        # Sort by latest creation date
        agent_summary.sort(key=lambda x: x["latest_created"] or "", reverse=True)
        
        return json.dumps({
            "status": "success",
            "registry_status": {
                "total_agents": total_agents,
                "total_versions": total_versions,
                "registry_file": str(_versions_base_path / "registry.json"),
                "versions_directory": str(_versions_base_path)
            },
            "agents": agent_summary,
            "message": f"Registry contains {total_versions} versions across {total_agents} agents"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get registry status: {str(e)}"
        })


@tool
def cleanup_old_versions(agent_name: str = "", keep_versions: int = 5) -> str:
    """
    Clean up old versions, keeping only the most recent ones.
    
    Args:
        agent_name: Specific agent to clean up (empty for all agents)
        keep_versions: Number of versions to keep per agent
    
    Returns:
        JSON string with cleanup result
    """
    try:
        cleaned_count = 0
        agents_to_clean = [agent_name] if agent_name else list(_version_registry.keys())
        
        for agent in agents_to_clean:
            if agent not in _version_registry:
                continue
            
            versions = _version_registry[agent]
            if len(versions) <= keep_versions:
                continue
            
            # Sort by creation date and keep only the most recent
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            versions_to_remove = versions[keep_versions:]
            
            # Remove old version files and directories
            for version_info in versions_to_remove:
                try:
                    version_file = Path(version_info["file_path"])
                    if version_file.exists():
                        version_file.unlink()
                    
                    # Remove version directory if empty
                    version_dir = version_file.parent
                    if version_dir.exists() and not any(version_dir.iterdir()):
                        version_dir.rmdir()
                    
                    cleaned_count += 1
                except Exception as e:
                    print(f"Failed to remove version file {version_info['file_path']}: {e}")
            
            # Update registry
            _version_registry[agent] = versions[:keep_versions]
        
        _save_version_registry()
        
        return json.dumps({
            "status": "success",
            "cleaned_versions": cleaned_count,
            "keep_versions": keep_versions,
            "message": f"Cleaned up {cleaned_count} old versions"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to cleanup versions: {str(e)}"
        })