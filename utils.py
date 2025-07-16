import sqlite3
import importlib.util
import sys
import string
import secrets
import traceback
import json
import os
import logging
from typing import Optional, List, Dict, Any

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Adjust level as needed

# Delay import of SqliteSaver to avoid potential metaclass conflicts during initial module loading
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
from langgraph.checkpoint.sqlite import SqliteSaver  # Import just before use

checkpointer = SqliteSaver(conn)

# --- Manifest Loading ---
_tools_manifest_cache: Optional[List[Dict[str, Any]]] = None
_agents_manifest_cache: Optional[List[Dict[str, Any]]] = None
_loaded_modules_cache: Dict[str, Any] = {}  # Cache for loaded modules


def _load_manifest(manifest_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads a JSON manifest file."""
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        if not isinstance(manifest_data, list):
            logger.error(
                f"Invalid manifest format in {manifest_path}. Expected a list."
            )
            return None
        # Basic validation (can be expanded)
        for entry in manifest_data:
            if not all(
                k in entry
                for k in ["name", "module_path", "function_name", "description"]
            ):
                logger.warning(
                    f"Manifest entry missing required keys in {manifest_path}: {entry}"
                )
                # Decide whether to skip or raise error
        return manifest_data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {manifest_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading manifest {manifest_path}: {e}", exc_info=True)
        return None


def _write_manifest(manifest_path: str, manifest_data: List[Dict[str, Any]]) -> bool:
    """Writes the manifest data back to the JSON file."""
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)
        logger.info(f"Successfully updated manifest file: {manifest_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing manifest {manifest_path}: {e}", exc_info=True)
        return False


def add_manifest_entry(manifest_type: str, entry: Dict[str, Any]) -> bool:
    """Adds or updates an entry in the specified manifest file."""
    if manifest_type not in ["agent", "tool"]:
        logger.error(f"Invalid manifest type specified: {manifest_type}")
        return False

    manifest_path = (
        "agents_manifest.json" if manifest_type == "agent" else "tools_manifest.json"
    )
    manifest_cache_attr = (
        "_agents_manifest_cache"
        if manifest_type == "agent"
        else "_tools_manifest_cache"
    )

    # Load current manifest (use internal function to bypass cache if needed, but loading normally should be fine)
    current_manifest = _load_manifest(manifest_path)
    if current_manifest is None:
        logger.error(f"Failed to load {manifest_path} to add entry.")
        # Optionally create an empty manifest if it doesn't exist?
        # For now, assume it should exist or fail.
        current_manifest = []  # Start fresh if load failed but we want to proceed

    entry_name = entry.get("name")
    if not entry_name:
        logger.error("Manifest entry must have a 'name'.")
        return False

    # Check if entry already exists and update it, otherwise append
    entry_found = False
    for i, existing_entry in enumerate(current_manifest):
        if existing_entry.get("name") == entry_name:
            current_manifest[i] = entry  # Update existing entry
            entry_found = True
            logger.info(
                f"Updating existing entry for '{entry_name}' in {manifest_path}."
            )
            break

    if not entry_found:
        current_manifest.append(entry)
        logger.info(f"Adding new entry for '{entry_name}' to {manifest_path}.")

    # Write the updated manifest back to disk
    if _write_manifest(manifest_path, current_manifest):
        # Clear the cache for this manifest type in utils.py
        globals()[manifest_cache_attr] = None
        logger.info(f"Cleared {manifest_cache_attr} cache.")
        # Also clear the specific module cache if the module path exists
        if "module_path" in entry:
            clear_module_cache(entry["module_path"])
        return True
    else:
        logger.error(f"Failed to write updated {manifest_path}.")
        return False


def get_tools_manifest() -> List[Dict[str, Any]]:
    """Loads and caches the tools manifest."""
    global _tools_manifest_cache
    if _tools_manifest_cache is None:
        # Use relative path from working directory (/app inside container)
        _tools_manifest_cache = _load_manifest("tools_manifest.json") or []
    return _tools_manifest_cache


def get_agents_manifest() -> List[Dict[str, Any]]:
    """Loads and caches the agents manifest."""
    global _agents_manifest_cache
    if _agents_manifest_cache is None:
        # Use relative path from working directory (/app inside container)
        _agents_manifest_cache = _load_manifest("agents_manifest.json") or []
    return _agents_manifest_cache


# --- Tool/Agent Discovery (using Manifest) ---


def all_tool_functions() -> list:
    """
    Loads and returns a list of all available tool functions based on the manifest.

    Returns:
        list: A list of callable tool functions.
    """
    manifest = get_tools_manifest()
    tool_funcs = []
    for tool_info in manifest:
        try:
            # Use load_registered_module to get the function directly
            tool_func = load_registered_module(tool_info)
            if tool_func:
                tool_funcs.append(tool_func)
        except Exception as e:
            # Log error if loading fails for a registered tool
            logger.error(
                f"Failed to load registered tool '{tool_info.get('name', 'unknown')}': {e}",
                exc_info=True,
            )
    return tool_funcs


def list_tools() -> list[str]:
    """Lists the names of all tools defined in the tools manifest."""
    manifest = get_tools_manifest()
    return [tool.get("name", "unnamed_tool") for tool in manifest]


def get_tool_details(name: str) -> Optional[Dict[str, Any]]:
    """Gets the manifest details for a specific tool by name."""
    manifest = get_tools_manifest()
    for tool_info in manifest:
        if tool_info.get("name") == name:
            return tool_info
    return None


def all_agents(exclude: list[str] = ["hermes"]) -> dict:
    """
    Returns available agents' names and descriptions from the agents manifest.

    Args:
        exclude (list[str], optional): A list of agent names to exclude.
                                       Defaults to ["hermes"].

    Returns:
        dict: A dictionary mapping agent names (str) to their descriptions (str).
    """
    manifest = get_agents_manifest()
    agent_details = {}
    for agent_info in manifest:
        agent_name = agent_info.get("name")
        if agent_name and agent_name not in exclude:
            agent_details[agent_name] = agent_info.get(
                "description", "No description available."
            )
    return agent_details


def list_agents() -> list[str]:
    """Lists the names of all agents defined in the agents manifest."""
    manifest = get_agents_manifest()
    return [agent.get("name", "unnamed_agent") for agent in manifest]


def get_agent_details(name: str) -> Optional[Dict[str, Any]]:
    """Gets the manifest details for a specific agent by name."""
    manifest = get_agents_manifest()
    for agent_info in manifest:
        if agent_info.get("name") == name:
            return agent_info
    return None


# --- Module Loading (Refactored) ---


def gensym(length: int = 32, prefix: str = "gensym_") -> str:
    """Generates a unique symbol (unchanged)."""
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    symbol = "".join([secrets.choice(alphabet) for i in range(length)])
    return prefix + symbol


def load_module(source: str, module_name: Optional[str] = None) -> Optional[Any]:
    """
    Dynamically loads a Python module from a source file path, using a cache.

    Args:
        source (str): The file path of the Python module to load.
        module_name (Optional[str], optional): The name to register the module
            under in sys.modules. If None, a unique name is generated.
            Defaults to None.

    Returns:
        Optional[module]: The loaded Python module object, or None if loading fails.
    """
    global _loaded_modules_cache

    # Use absolute path for consistent caching
    abs_source = os.path.abspath(source)

    if abs_source in _loaded_modules_cache:
        # logger.debug(f"Returning cached module for: {abs_source}")
        return _loaded_modules_cache[abs_source]

    if not os.path.exists(abs_source):
        logger.error(f"Module source file not found: {abs_source}")
        return None

    if module_name is None:
        # Use a more descriptive name based on the file if possible
        base_name = os.path.splitext(os.path.basename(abs_source))[0]
        module_name = f"loaded_{base_name}_{gensym(8, '')}"  # Shorter random part

    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_source)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec for: {abs_source}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Register before execution
        spec.loader.exec_module(module)
        logger.info(f"Successfully loaded module '{module_name}' from: {abs_source}")
        _loaded_modules_cache[abs_source] = module  # Cache on success
        return module
    except Exception as e:
        logger.error(
            f"Failed to load module '{module_name}' from {abs_source}: {e}",
            exc_info=True,
        )
        # Remove from sys.modules if loading failed
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None


def load_registered_module(manifest_entry: Dict[str, Any]) -> Optional[callable]:
    """
    Loads a module based on a manifest entry and returns the specified function.

    Args:
        manifest_entry (Dict[str, Any]): A dictionary from the manifest file.

    Returns:
        Optional[callable]: The callable function/object specified in the manifest,
                           or None if loading or attribute access fails.
    """
    module_path = manifest_entry.get("module_path")
    function_name = manifest_entry.get("function_name")
    entry_name = manifest_entry.get("name", "unknown")  # For logging

    if not module_path or not function_name:
        logger.error(
            f"Manifest entry '{entry_name}' is missing 'module_path' or 'function_name'."
        )
        return None

    module = load_module(module_path)  # Use the cached loader

    if module is None:
        logger.error(f"Failed to load module '{module_path}' for entry '{entry_name}'.")
        return None

    try:
        func = getattr(module, function_name)
        if not callable(func):
            logger.error(
                f"Attribute '{function_name}' in module '{module_path}' for entry '{entry_name}' is not callable."
            )
            return None
        return func
    except AttributeError:
        logger.error(
            f"Function '{function_name}' not found in module '{module_path}' for entry '{entry_name}'."
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error getting function '{function_name}' from module '{module_path}' for entry '{entry_name}': {e}",
            exc_info=True,
        )
        return None


def clear_module_cache(file_path: str) -> None:
    """Removes a specific module from the loaded module cache."""
    global _loaded_modules_cache
    abs_path = os.path.abspath(file_path)
    if abs_path in _loaded_modules_cache:
        module_name = _loaded_modules_cache[abs_path].__name__
        del _loaded_modules_cache[abs_path]
        if module_name in sys.modules:
            del sys.modules[module_name]  # Also remove from sys.modules
        logger.info(
            f"Cleared module cache and sys.modules entry for: {abs_path} (module name: {module_name})"
        )
    else:
        logger.debug(f"Module path not found in cache, no need to clear: {abs_path}")


# --- Deprecated/Removed Functions ---
# list_broken_tools, list_broken_agents are removed as loading errors are now logged directly.
