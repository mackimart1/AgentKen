from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import datetime


class SaveContextInput(BaseModel):
    session_id: str = Field(description="Session identifier")
    context_type: str = Field(description="Type of context: preference, pattern, insight, goal")
    data: Dict[str, Any] = Field(description="Context data to save")
    importance: int = Field(default=5, description="Importance level (1-10)")


class LoadContextInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    context_type: Optional[str] = Field(default=None, description="Filter by context type")
    limit: int = Field(default=10, description="Maximum number of context items to return")


class UpdatePreferenceInput(BaseModel):
    preference_key: str = Field(description="Preference key")
    preference_value: Any = Field(description="Preference value")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


# Global context storage (in production, this would be integrated with memory_manager)
_context_storage: Dict[str, List[Dict[str, Any]]] = {}
_user_preferences: Dict[str, Dict[str, Any]] = {}


@tool(args_schema=SaveContextInput)
def save_context(
    session_id: str,
    context_type: str,
    data: Dict[str, Any],
    importance: int = 5
) -> str:
    """
    Save context information for future sessions.
    
    Args:
        session_id: Session identifier
        context_type: Type of context (preference, pattern, insight, goal)
        data: Context data to save
        importance: Importance level (1-10)
    
    Returns:
        JSON string with save result
    """
    try:
        if session_id not in _context_storage:
            _context_storage[session_id] = []
        
        context_item = {
            "id": f"ctx_{len(_context_storage[session_id])}_{datetime.datetime.now().timestamp()}",
            "session_id": session_id,
            "context_type": context_type,
            "data": data,
            "importance": importance,
            "created_at": datetime.datetime.now().isoformat(),
            "accessed_count": 0,
            "last_accessed": None
        }
        
        _context_storage[session_id].append(context_item)
        
        # Also save to memory manager if available
        try:
            import memory_manager
            memory_manager_instance = memory_manager.MemoryManager()
            
            memory_manager_instance.add_memory(
                key=context_item["id"],
                value=json.dumps(context_item),
                memory_type="context",
                agent_name="hermes",
                importance=importance
            )
        except Exception as e:
            # Continue even if memory manager fails
            pass
        
        return json.dumps({
            "status": "success",
            "context_id": context_item["id"],
            "message": f"Context saved successfully: {context_type}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to save context: {str(e)}"
        })


@tool(args_schema=LoadContextInput)
def load_context(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    context_type: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Load context information from previous sessions.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        context_type: Filter by context type
        limit: Maximum number of context items to return
    
    Returns:
        JSON string with context data
    """
    try:
        all_contexts = []
        
        # Collect contexts from storage
        for sid, contexts in _context_storage.items():
            if session_id and sid != session_id:
                continue
            
            for context in contexts:
                if context_type and context["context_type"] != context_type:
                    continue
                
                # Update access tracking
                context["accessed_count"] += 1
                context["last_accessed"] = datetime.datetime.now().isoformat()
                
                all_contexts.append(context)
        
        # Try to load from memory manager as well
        try:
            import memory_manager
            memory_manager_instance = memory_manager.MemoryManager()
            
            memories = memory_manager_instance.retrieve_memories(
                memory_type="context",
                agent_name="hermes",
                limit=limit * 2  # Get more to filter
            )
            
            for memory in memories:
                try:
                    memory_data = json.loads(memory.get("value", "{}"))
                    if isinstance(memory_data, dict) and "context_type" in memory_data:
                        if context_type and memory_data["context_type"] != context_type:
                            continue
                        all_contexts.append(memory_data)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            # Continue even if memory manager fails
            pass
        
        # Sort by importance and recency
        all_contexts.sort(key=lambda x: (
            -x.get("importance", 0),
            -datetime.datetime.fromisoformat(x.get("created_at", "1970-01-01")).timestamp()
        ))
        
        # Limit results
        limited_contexts = all_contexts[:limit]
        
        return json.dumps({
            "status": "success",
            "contexts": limited_contexts,
            "count": len(limited_contexts),
            "message": f"Loaded {len(limited_contexts)} context items"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to load context: {str(e)}"
        })


@tool(args_schema=UpdatePreferenceInput)
def update_user_preference(
    preference_key: str,
    preference_value: Any,
    session_id: Optional[str] = None
) -> str:
    """
    Update user preferences for future sessions.
    
    Args:
        preference_key: Preference key
        preference_value: Preference value
        session_id: Session identifier
    
    Returns:
        JSON string with update result
    """
    try:
        user_key = session_id or "default_user"
        
        if user_key not in _user_preferences:
            _user_preferences[user_key] = {}
        
        _user_preferences[user_key][preference_key] = {
            "value": preference_value,
            "updated_at": datetime.datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Also save as context
        context_data = {
            "preference_key": preference_key,
            "preference_value": preference_value,
            "type": "preference_update"
        }
        
        if session_id:
            save_context(session_id, "preference", context_data, importance=7)
        
        return json.dumps({
            "status": "success",
            "message": f"Preference updated: {preference_key}"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to update preference: {str(e)}"
        })


@tool
def get_user_preferences(session_id: Optional[str] = None) -> str:
    """
    Get user preferences for the current session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        JSON string with user preferences
    """
    try:
        user_key = session_id or "default_user"
        preferences = _user_preferences.get(user_key, {})
        
        # Extract just the values
        simplified_preferences = {
            key: pref["value"] for key, pref in preferences.items()
        }
        
        return json.dumps({
            "status": "success",
            "preferences": simplified_preferences,
            "count": len(simplified_preferences),
            "message": f"Retrieved {len(simplified_preferences)} preferences"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get preferences: {str(e)}"
        })


@tool
def analyze_context_patterns() -> str:
    """
    Analyze context patterns to identify insights and trends.
    
    Returns:
        JSON string with pattern analysis
    """
    try:
        analysis = {
            "total_contexts": 0,
            "context_types": {},
            "frequent_patterns": [],
            "success_indicators": [],
            "failure_indicators": [],
            "recommendations": []
        }
        
        # Analyze all stored contexts
        for session_contexts in _context_storage.values():
            analysis["total_contexts"] += len(session_contexts)
            
            for context in session_contexts:
                context_type = context.get("context_type", "unknown")
                analysis["context_types"][context_type] = analysis["context_types"].get(context_type, 0) + 1
                
                # Look for patterns in the data
                data = context.get("data", {})
                if isinstance(data, dict):
                    if data.get("success", False):
                        pattern = data.get("pattern", "")
                        if pattern and pattern not in analysis["success_indicators"]:
                            analysis["success_indicators"].append(pattern)
                    elif data.get("success") is False:
                        pattern = data.get("pattern", "")
                        if pattern and pattern not in analysis["failure_indicators"]:
                            analysis["failure_indicators"].append(pattern)
        
        # Generate recommendations
        if analysis["success_indicators"]:
            analysis["recommendations"].append("Leverage successful patterns: " + ", ".join(analysis["success_indicators"][:3]))
        
        if analysis["failure_indicators"]:
            analysis["recommendations"].append("Avoid failure patterns: " + ", ".join(analysis["failure_indicators"][:3]))
        
        if analysis["context_types"].get("preference", 0) > 5:
            analysis["recommendations"].append("Strong preference learning detected - use for personalization")
        
        return json.dumps({
            "status": "success",
            "analysis": analysis,
            "message": "Context pattern analysis completed"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to analyze context patterns: {str(e)}"
        })


@tool
def clear_old_context(days_old: int = 30) -> str:
    """
    Clear context data older than specified days.
    
    Args:
        days_old: Number of days old to consider for deletion
    
    Returns:
        JSON string with cleanup result
    """
    try:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        cleared_count = 0
        
        for session_id in list(_context_storage.keys()):
            contexts = _context_storage[session_id]
            original_count = len(contexts)
            
            # Filter out old contexts
            _context_storage[session_id] = [
                context for context in contexts
                if datetime.datetime.fromisoformat(context.get("created_at", "1970-01-01")) > cutoff_date
            ]
            
            cleared_count += original_count - len(_context_storage[session_id])
            
            # Remove empty sessions
            if not _context_storage[session_id]:
                del _context_storage[session_id]
        
        return json.dumps({
            "status": "success",
            "cleared_count": cleared_count,
            "message": f"Cleared {cleared_count} old context items"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to clear old context: {str(e)}"
        })