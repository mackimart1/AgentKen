from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class ToolStatus(Enum):
    """Tool lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    OBSOLETE = "obsolete"
    EXPERIMENTAL = "experimental"
    BETA = "beta"


class DeprecateToolInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to deprecate")
    reason: str = Field(description="Reason for deprecation")
    replacement_tool: str = Field(default="", description="Name of replacement tool")
    deprecation_period_days: int = Field(default=90, description="Days until removal")


class UpdateToolStatusInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to update")
    new_status: str = Field(description="New status: active, deprecated, obsolete, experimental, beta")
    reason: str = Field(default="", description="Reason for status change")


class RemoveToolInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to remove")
    force: bool = Field(default=False, description="Force removal even if not ready")
    backup: bool = Field(default=True, description="Create backup before removal")


class AnalyzeUsageInput(BaseModel):
    days_threshold: int = Field(default=30, description="Days to consider for usage analysis")
    include_stats: bool = Field(default=True, description="Include detailed statistics")


# Global tool metadata storage
_tool_metadata: Dict[str, Dict[str, Any]] = {}
_metadata_file = Path("tools_metadata.json")
_backup_dir = Path("tools_backup")
_backup_dir.mkdir(exist_ok=True)


def _load_tool_metadata():
    """Load tool metadata from file."""
    global _tool_metadata
    
    if _metadata_file.exists():
        try:
            with open(_metadata_file, 'r') as f:
                _tool_metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load tool metadata: {e}")
            _tool_metadata = {}
    else:
        _tool_metadata = {}


def _save_tool_metadata():
    """Save tool metadata to file."""
    try:
        with open(_metadata_file, 'w') as f:
            json.dump(_tool_metadata, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save tool metadata: {e}")


def _initialize_tool_metadata(tool_name: str) -> Dict[str, Any]:
    """Initialize metadata for a tool if it doesn't exist."""
    if tool_name not in _tool_metadata:
        _tool_metadata[tool_name] = {
            "name": tool_name,
            "status": ToolStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "author": "ToolMaker",
            "description": "",
            "category": "general",
            "usage_count": 0,
            "last_used": None,
            "deprecation_date": None,
            "removal_date": None,
            "deprecation_reason": "",
            "replacement_tool": "",
            "dependencies": [],
            "file_size": 0,
            "complexity_score": 0
        }
        _save_tool_metadata()
    
    return _tool_metadata[tool_name]


def _update_tool_file_info(tool_name: str):
    """Update file information for a tool."""
    tool_file = Path(f"tools/{tool_name}.py")
    
    if tool_file.exists():
        stat = tool_file.stat()
        metadata = _tool_metadata.get(tool_name, {})
        metadata["file_size"] = stat.st_size
        metadata["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Calculate complexity score (simple line count based)
        try:
            with open(tool_file, 'r') as f:
                lines = f.readlines()
            
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            metadata["complexity_score"] = min(100, code_lines / 2)
        except Exception:
            metadata["complexity_score"] = 0
        
        _tool_metadata[tool_name] = metadata
        _save_tool_metadata()


def _create_tool_backup(tool_name: str) -> str:
    """Create backup of tool files."""
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tool_backup_dir = _backup_dir / f"{tool_name}_{backup_timestamp}"
    tool_backup_dir.mkdir(exist_ok=True)
    
    files_backed_up = []
    
    # Backup main tool file
    tool_file = Path(f"tools/{tool_name}.py")
    if tool_file.exists():
        backup_file = tool_backup_dir / f"{tool_name}.py"
        shutil.copy2(tool_file, backup_file)
        files_backed_up.append(str(backup_file))
    
    # Backup test file
    test_file = Path(f"tests/tools/test_{tool_name}.py")
    if test_file.exists():
        backup_test = tool_backup_dir / f"test_{tool_name}.py"
        shutil.copy2(test_file, backup_test)
        files_backed_up.append(str(backup_test))
    
    # Backup documentation
    doc_file = Path(f"docs/tools/{tool_name}.md")
    if doc_file.exists():
        backup_doc = tool_backup_dir / f"{tool_name}.md"
        shutil.copy2(doc_file, backup_doc)
        files_backed_up.append(str(backup_doc))
    
    # Save backup metadata
    backup_info = {
        "tool_name": tool_name,
        "backup_date": datetime.now().isoformat(),
        "files_backed_up": files_backed_up,
        "metadata": _tool_metadata.get(tool_name, {}).copy()
    }
    
    backup_info_file = tool_backup_dir / "backup_info.json"
    with open(backup_info_file, 'w') as f:
        json.dump(backup_info, f, indent=2)
    
    return str(tool_backup_dir)


# Load metadata on module import
_load_tool_metadata()


@tool(args_schema=DeprecateToolInput)
def deprecate_tool(
    tool_name: str,
    reason: str,
    replacement_tool: str = "",
    deprecation_period_days: int = 90
) -> str:
    """
    Mark a tool as deprecated with a removal timeline.
    
    Args:
        tool_name: Name of the tool to deprecate
        reason: Reason for deprecation
        replacement_tool: Name of replacement tool (if any)
        deprecation_period_days: Days until removal (default 90)
    
    Returns:
        JSON string with deprecation result
    """
    try:
        # Initialize metadata if needed
        metadata = _initialize_tool_metadata(tool_name)
        
        # Check if tool exists
        tool_file = Path(f"tools/{tool_name}.py")
        if not tool_file.exists():
            return json.dumps({
                "status": "failure",
                "message": f"Tool file not found: {tool_name}"
            })
        
        # Update metadata
        metadata["status"] = ToolStatus.DEPRECATED.value
        metadata["deprecation_date"] = datetime.now().isoformat()
        metadata["removal_date"] = (datetime.now() + timedelta(days=deprecation_period_days)).isoformat()
        metadata["deprecation_reason"] = reason
        metadata["replacement_tool"] = replacement_tool
        metadata["updated_at"] = datetime.now().isoformat()
        
        _tool_metadata[tool_name] = metadata
        _save_tool_metadata()
        
        # Create backup
        backup_path = _create_tool_backup(tool_name)
        
        # Add deprecation warning to tool file
        _add_deprecation_warning(tool_name, reason, replacement_tool, metadata["removal_date"])
        
        return json.dumps({
            "status": "success",
            "tool_name": tool_name,
            "deprecation_date": metadata["deprecation_date"],
            "removal_date": metadata["removal_date"],
            "reason": reason,
            "replacement_tool": replacement_tool,
            "backup_path": backup_path,
            "message": f"Tool {tool_name} marked as deprecated. Removal scheduled for {metadata['removal_date'][:10]}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to deprecate tool: {str(e)}"
        })


def _add_deprecation_warning(tool_name: str, reason: str, replacement_tool: str, removal_date: str):
    """Add deprecation warning to tool file."""
    tool_file = Path(f"tools/{tool_name}.py")
    
    try:
        with open(tool_file, 'r') as f:
            content = f.read()
        
        # Create deprecation warning
        warning = f'''"""
⚠️  DEPRECATION WARNING ⚠️

This tool is deprecated and will be removed on {removal_date[:10]}.

Reason: {reason}
Replacement: {replacement_tool if replacement_tool else "None specified"}

Please update your code to use the replacement tool or alternative solution.
"""

import warnings
warnings.warn(
    f"Tool '{tool_name}' is deprecated and will be removed on {removal_date[:10]}. "
    f"Reason: {reason}. Replacement: {replacement_tool if replacement_tool else 'None specified'}",
    DeprecationWarning,
    stacklevel=2
)

'''
        
        # Insert warning after imports but before tool definition
        lines = content.split('\n')
        insert_index = 0
        
        # Find a good place to insert (after imports, before @tool)
        for i, line in enumerate(lines):
            if line.strip().startswith('@tool') or line.strip().startswith('def '):
                insert_index = i
                break
            elif line.strip() and not (line.strip().startswith('from ') or 
                                     line.strip().startswith('import ') or
                                     line.strip().startswith('#') or
                                     line.strip().startswith('"""') or
                                     line.strip().startswith("'''")):
                insert_index = i
                break
        
        # Insert warning
        lines.insert(insert_index, warning)
        
        # Write back to file
        with open(tool_file, 'w') as f:
            f.write('\n'.join(lines))
            
    except Exception as e:
        print(f"Warning: Failed to add deprecation warning to {tool_name}: {e}")


@tool(args_schema=UpdateToolStatusInput)
def update_tool_status(tool_name: str, new_status: str, reason: str = "") -> str:
    """
    Update the status of a tool in the lifecycle system.
    
    Args:
        tool_name: Name of the tool to update
        new_status: New status (active, deprecated, obsolete, experimental, beta)
        reason: Reason for status change
    
    Returns:
        JSON string with update result
    """
    try:
        # Validate status
        try:
            status_enum = ToolStatus(new_status.lower())
        except ValueError:
            return json.dumps({
                "status": "failure",
                "message": f"Invalid status: {new_status}. Valid options: {[s.value for s in ToolStatus]}"
            })
        
        # Initialize metadata if needed
        metadata = _initialize_tool_metadata(tool_name)
        
        old_status = metadata["status"]
        metadata["status"] = status_enum.value
        metadata["updated_at"] = datetime.now().isoformat()
        
        if reason:
            metadata["status_change_reason"] = reason
        
        # Special handling for status changes
        if status_enum == ToolStatus.ACTIVE and old_status == ToolStatus.DEPRECATED.value:
            # Reactivating deprecated tool
            metadata["deprecation_date"] = None
            metadata["removal_date"] = None
            metadata["deprecation_reason"] = ""
        
        _tool_metadata[tool_name] = metadata
        _save_tool_metadata()
        
        return json.dumps({
            "status": "success",
            "tool_name": tool_name,
            "old_status": old_status,
            "new_status": status_enum.value,
            "reason": reason,
            "updated_at": metadata["updated_at"],
            "message": f"Tool {tool_name} status updated from {old_status} to {status_enum.value}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to update tool status: {str(e)}"
        })


@tool(args_schema=RemoveToolInput)
def remove_tool(tool_name: str, force: bool = False, backup: bool = True) -> str:
    """
    Remove a tool from the system (only if deprecated and past removal date).
    
    Args:
        tool_name: Name of the tool to remove
        force: Force removal even if not ready
        backup: Create backup before removal
    
    Returns:
        JSON string with removal result
    """
    try:
        if tool_name not in _tool_metadata:
            return json.dumps({
                "status": "failure",
                "message": f"Tool metadata not found: {tool_name}"
            })
        
        metadata = _tool_metadata[tool_name]
        
        # Check if tool is ready for removal
        if not force:
            if metadata["status"] != ToolStatus.DEPRECATED.value:
                return json.dumps({
                    "status": "failure",
                    "message": f"Tool {tool_name} is not deprecated. Current status: {metadata['status']}"
                })
            
            if metadata.get("removal_date"):
                removal_date = datetime.fromisoformat(metadata["removal_date"])
                if datetime.now() < removal_date:
                    return json.dumps({
                        "status": "failure",
                        "message": f"Tool {tool_name} is not yet ready for removal. Removal date: {removal_date.strftime('%Y-%m-%d')}"
                    })
        
        removed_files = []
        backup_path = None
        
        # Create backup if requested
        if backup:
            backup_path = _create_tool_backup(tool_name)
        
        # Remove tool file
        tool_file = Path(f"tools/{tool_name}.py")
        if tool_file.exists():
            tool_file.unlink()
            removed_files.append(str(tool_file))
        
        # Remove test file
        test_file = Path(f"tests/tools/test_{tool_name}.py")
        if test_file.exists():
            test_file.unlink()
            removed_files.append(str(test_file))
        
        # Remove documentation
        doc_file = Path(f"docs/tools/{tool_name}.md")
        if doc_file.exists():
            doc_file.unlink()
            removed_files.append(str(doc_file))
        
        # Update metadata to mark as removed
        metadata["status"] = ToolStatus.OBSOLETE.value
        metadata["removed_at"] = datetime.now().isoformat()
        metadata["removed_files"] = removed_files
        
        if backup_path:
            metadata["backup_path"] = backup_path
        
        _tool_metadata[tool_name] = metadata
        _save_tool_metadata()
        
        return json.dumps({
            "status": "success",
            "tool_name": tool_name,
            "removed_files": removed_files,
            "backup_path": backup_path,
            "removed_at": metadata["removed_at"],
            "message": f"Tool {tool_name} successfully removed from system"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to remove tool: {str(e)}"
        })


@tool(args_schema=AnalyzeUsageInput)
def analyze_tool_usage(days_threshold: int = 30, include_stats: bool = True) -> str:
    """
    Analyze tool usage patterns and identify unused or underused tools.
    
    Args:
        days_threshold: Days to consider for usage analysis
        include_stats: Include detailed statistics
    
    Returns:
        JSON string with usage analysis
    """
    try:
        # Update file info for all tools
        for tool_name in _tool_metadata.keys():
            _update_tool_file_info(tool_name)
        
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "days_threshold": days_threshold,
            "total_tools": len(_tool_metadata),
            "by_status": {},
            "unused_tools": [],
            "deprecated_tools": [],
            "tools_for_removal": [],
            "active_tools": [],
            "statistics": {} if include_stats else None
        }
        
        # Analyze by status
        for tool_name, metadata in _tool_metadata.items():
            status = metadata["status"]
            analysis["by_status"][status] = analysis["by_status"].get(status, 0) + 1
            
            # Check usage patterns
            last_used = metadata.get("last_used")
            if last_used:
                last_used_date = datetime.fromisoformat(last_used)
                is_recently_used = last_used_date > threshold_date
            else:
                is_recently_used = False
            
            # Categorize tools
            if status == ToolStatus.ACTIVE.value:
                if not is_recently_used:
                    analysis["unused_tools"].append({
                        "name": tool_name,
                        "last_used": last_used,
                        "usage_count": metadata.get("usage_count", 0),
                        "created_at": metadata.get("created_at"),
                        "file_size": metadata.get("file_size", 0)
                    })
                else:
                    analysis["active_tools"].append(tool_name)
            
            elif status == ToolStatus.DEPRECATED.value:
                analysis["deprecated_tools"].append({
                    "name": tool_name,
                    "deprecation_date": metadata.get("deprecation_date"),
                    "removal_date": metadata.get("removal_date"),
                    "reason": metadata.get("deprecation_reason", ""),
                    "replacement": metadata.get("replacement_tool", "")
                })
                
                # Check if ready for removal
                removal_date = metadata.get("removal_date")
                if removal_date:
                    removal_datetime = datetime.fromisoformat(removal_date)
                    if datetime.now() >= removal_datetime:
                        analysis["tools_for_removal"].append(tool_name)
        
        # Generate statistics
        if include_stats:
            total_usage = sum(metadata.get("usage_count", 0) for metadata in _tool_metadata.values())
            total_size = sum(metadata.get("file_size", 0) for metadata in _tool_metadata.values())
            
            analysis["statistics"] = {
                "total_usage_count": total_usage,
                "average_usage_per_tool": total_usage / len(_tool_metadata) if _tool_metadata else 0,
                "total_file_size_bytes": total_size,
                "average_file_size_bytes": total_size / len(_tool_metadata) if _tool_metadata else 0,
                "unused_tool_count": len(analysis["unused_tools"]),
                "unused_percentage": len(analysis["unused_tools"]) / len(_tool_metadata) * 100 if _tool_metadata else 0,
                "deprecated_count": len(analysis["deprecated_tools"]),
                "ready_for_removal_count": len(analysis["tools_for_removal"])
            }
        
        # Generate recommendations
        recommendations = []
        
        if len(analysis["unused_tools"]) > 0:
            recommendations.append(f"Consider deprecating {len(analysis['unused_tools'])} unused tools")
        
        if len(analysis["tools_for_removal"]) > 0:
            recommendations.append(f"Remove {len(analysis['tools_for_removal'])} tools that are past their removal date")
        
        if analysis.get("statistics", {}).get("unused_percentage", 0) > 30:
            recommendations.append("High percentage of unused tools - consider cleanup")
        
        analysis["recommendations"] = recommendations
        
        return json.dumps({
            "status": "success",
            "analysis": analysis,
            "message": f"Usage analysis completed for {len(_tool_metadata)} tools"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Usage analysis failed: {str(e)}"
        })


@tool
def get_tool_lifecycle_status() -> str:
    """
    Get comprehensive lifecycle status for all tools.
    
    Returns:
        JSON string with lifecycle status
    """
    try:
        status_summary = {
            "total_tools": len(_tool_metadata),
            "by_status": {},
            "recent_changes": [],
            "upcoming_removals": [],
            "system_health": {}
        }
        
        # Count by status
        for metadata in _tool_metadata.values():
            status = metadata["status"]
            status_summary["by_status"][status] = status_summary["by_status"].get(status, 0) + 1
        
        # Find recent changes (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        
        for tool_name, metadata in _tool_metadata.items():
            updated_at = datetime.fromisoformat(metadata.get("updated_at", metadata.get("created_at", "1970-01-01")))
            
            if updated_at > week_ago:
                status_summary["recent_changes"].append({
                    "tool_name": tool_name,
                    "status": metadata["status"],
                    "updated_at": metadata["updated_at"],
                    "change_type": "status_change" if "status_change_reason" in metadata else "general_update"
                })
        
        # Find upcoming removals (next 30 days)
        month_ahead = datetime.now() + timedelta(days=30)
        
        for tool_name, metadata in _tool_metadata.items():
            if metadata["status"] == ToolStatus.DEPRECATED.value and metadata.get("removal_date"):
                removal_date = datetime.fromisoformat(metadata["removal_date"])
                if removal_date <= month_ahead:
                    status_summary["upcoming_removals"].append({
                        "tool_name": tool_name,
                        "removal_date": metadata["removal_date"],
                        "days_until_removal": (removal_date - datetime.now()).days,
                        "reason": metadata.get("deprecation_reason", ""),
                        "replacement": metadata.get("replacement_tool", "")
                    })
        
        # System health metrics
        total_tools = len(_tool_metadata)
        active_tools = status_summary["by_status"].get(ToolStatus.ACTIVE.value, 0)
        deprecated_tools = status_summary["by_status"].get(ToolStatus.DEPRECATED.value, 0)
        
        status_summary["system_health"] = {
            "active_percentage": (active_tools / total_tools * 100) if total_tools > 0 else 0,
            "deprecated_percentage": (deprecated_tools / total_tools * 100) if total_tools > 0 else 0,
            "health_score": max(0, 100 - (deprecated_tools / total_tools * 50)) if total_tools > 0 else 100,
            "needs_attention": deprecated_tools > total_tools * 0.2  # More than 20% deprecated
        }
        
        return json.dumps({
            "status": "success",
            "lifecycle_status": status_summary,
            "metadata_file": str(_metadata_file),
            "backup_directory": str(_backup_dir),
            "message": f"Lifecycle status retrieved for {total_tools} tools"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get lifecycle status: {str(e)}"
        })


@tool
def cleanup_tool_lifecycle() -> str:
    """
    Clean up tool lifecycle data and old backups.
    
    Returns:
        JSON string with cleanup results
    """
    try:
        cleanup_results = {
            "removed_obsolete_metadata": 0,
            "cleaned_old_backups": 0,
            "updated_file_info": 0
        }
        
        # Remove metadata for obsolete tools (removed more than 30 days ago)
        cutoff_date = datetime.now() - timedelta(days=30)
        obsolete_tools = []
        
        for tool_name, metadata in _tool_metadata.items():
            if (metadata["status"] == ToolStatus.OBSOLETE.value and 
                metadata.get("removed_at")):
                removed_date = datetime.fromisoformat(metadata["removed_at"])
                if removed_date < cutoff_date:
                    obsolete_tools.append(tool_name)
        
        for tool_name in obsolete_tools:
            del _tool_metadata[tool_name]
            cleanup_results["removed_obsolete_metadata"] += 1
        
        # Clean old backups (older than 90 days)
        backup_cutoff = datetime.now() - timedelta(days=90)
        
        if _backup_dir.exists():
            for backup_item in _backup_dir.iterdir():
                if backup_item.is_dir():
                    try:
                        # Extract timestamp from directory name
                        timestamp_str = backup_item.name.split('_')[-2] + '_' + backup_item.name.split('_')[-1]
                        backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        if backup_date < backup_cutoff:
                            shutil.rmtree(backup_item)
                            cleanup_results["cleaned_old_backups"] += 1
                    except (ValueError, IndexError):
                        # Skip if can't parse date
                        continue
        
        # Update file info for existing tools
        for tool_name in _tool_metadata.keys():
            if _tool_metadata[tool_name]["status"] != ToolStatus.OBSOLETE.value:
                _update_tool_file_info(tool_name)
                cleanup_results["updated_file_info"] += 1
        
        # Save updated metadata
        _save_tool_metadata()
        
        return json.dumps({
            "status": "success",
            "cleanup_results": cleanup_results,
            "message": f"Lifecycle cleanup completed: {sum(cleanup_results.values())} items processed"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Lifecycle cleanup failed: {str(e)}"
        })