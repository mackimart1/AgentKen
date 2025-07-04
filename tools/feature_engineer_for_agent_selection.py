import json
from typing import Dict, List, Any
from langchain_core.tools import tool


@tool
def feature_engineer_for_agent_selection(
    task_description: str,
    agent_metadata: List[Dict[str, Any]],
    context_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extracts and structures features from task description, agent metadata,
    and runtime context for agent selection models.

    Args:
        task_description: The textual description of the task.
        agent_metadata: A list of dictionaries, where each dictionary contains
                        metadata about an agent (e.g., name, capabilities, type).
        context_info: A dictionary containing runtime context information
                      (e.g., agent load, queue length, system status).

    Returns:
        A dictionary containing structured features ready for further processing
        or direct use by an agent selection model. Includes 'task_features',
        'agent_features', and 'context_features'.
    """
    # Basic feature extraction/structuring
    # Task features could involve length, keywords (simple example), or the raw text for embedding
    task_features = {
        "description": task_description,
        "description_length": len(task_description),
        # In a more advanced version, add embeddings, keyword extraction, etc.
    }

    # Agent features could involve counts, types, or specific metadata points
    agent_features = {
        "count": len(agent_metadata),
        "metadata_list": agent_metadata,
        # In a more advanced version, aggregate features across agents
    }

    # Context features are typically used directly
    context_features = context_info

    # Combine into a single feature dictionary
    feature_vector = {
        "task_features": task_features,
        "agent_features": agent_features,
        "context_features": context_features,
    }

    # Ensure the output is serializable (e.g., if complex objects were involved)
    # For this basic version, it should be fine, but good practice.
    try:
        # Attempt to serialize to catch potential issues early
        json.dumps(feature_vector)
    except TypeError as e:
        raise ValueError(f"Feature vector is not JSON serializable: {e}")

    return feature_vector
