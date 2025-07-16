"""
AgentKen Modular System
Provides plug-and-play modularity for agents and tools with flexible composition,
easy upgrades, and seamless addition/removal of components.
"""

import json
import logging
import importlib
import inspect
import os
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union
import traceback
from collections import defaultdict


class ModuleType(Enum):
    """Types of modules in the system"""
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    EXTENSION = "extension"


class ModuleStatus(Enum):
    """Module lifecycle status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class DependencyType(Enum):
    """Types of module dependencies"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONFLICTING = "conflicting"


@dataclass
class ModuleDependency:
    """Represents a module dependency"""
    name: str
    version: Optional[str] = None
    dependency_type: DependencyType = DependencyType.REQUIRED
    description: str = ""


@dataclass
class ModuleCapability:
    """Describes what a module can do"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ModuleMetadata:
    """Comprehensive module metadata"""
    id: str
    name: str
    version: str
    module_type: ModuleType
    description: str
    author: str = ""
    license: str = ""
    homepage: str = ""
    documentation: str = ""
    
    # Dependencies and requirements
    dependencies: List[ModuleDependency] = field(default_factory=list)
    python_requirements: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    
    # Capabilities and interface
    capabilities: List[ModuleCapability] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle and compatibility
    min_agentken_version: str = "1.0.0"
    max_agentken_version: Optional[str] = None
    supported_platforms: List[str] = field(default_factory=lambda: ["any"])
    
    # Runtime information
    status: ModuleStatus = ModuleStatus.UNLOADED
    load_time: Optional[float] = None
    error_message: Optional[str] = None
    last_used: Optional[float] = None
    usage_count: int = 0
    
    # Module file information
    module_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Tags and categorization
    tags: List[str] = field(default_factory=list)
    category: str = ""
    priority: int = 0


class ModuleInterface(ABC):
    """Base interface that all modules must implement"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        self.module_id = module_id
        self.config = config or {}
        self.metadata: Optional[ModuleMetadata] = None
        self.is_initialized = False
        self.logger = logging.getLogger(f"module.{module_id}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the module. Return True if successful."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the module. Return True if successful."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ModuleCapability]:
        """Return list of capabilities this module provides."""
        pass
    
    @abstractmethod
    def execute(self, capability: str, **kwargs) -> Any:
        """Execute a specific capability."""
        pass
    
    def get_metadata(self) -> ModuleMetadata:
        """Get module metadata."""
        return self.metadata
    
    def get_status(self) -> ModuleStatus:
        """Get current module status."""
        return self.metadata.status if self.metadata else ModuleStatus.UNLOADED
    
    def get_health(self) -> Dict[str, Any]:
        """Get module health information."""
        return {
            "status": self.get_status().value,
            "is_initialized": self.is_initialized,
            "last_used": self.metadata.last_used if self.metadata else None,
            "usage_count": self.metadata.usage_count if self.metadata else 0,
            "error_message": self.metadata.error_message if self.metadata else None
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        # Basic validation - can be overridden by modules
        if not self.metadata or not self.metadata.configuration_schema:
            return True
        
        # TODO: Implement JSON schema validation
        return True
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update module configuration."""
        if self.validate_config(config):
            self.config.update(config)
            return True
        return False


class AgentModule(ModuleInterface):
    """Base class for agent modules"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        super().__init__(module_id, config)
        self.agent_instance = None
    
    @abstractmethod
    def create_agent(self) -> Any:
        """Create and return the agent instance."""
        pass
    
    def get_agent(self) -> Any:
        """Get the agent instance, creating it if necessary."""
        if not self.agent_instance and self.is_initialized:
            self.agent_instance = self.create_agent()
        return self.agent_instance


class ToolModule(ModuleInterface):
    """Base class for tool modules"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        super().__init__(module_id, config)
        self.tool_functions: Dict[str, Callable] = {}
    
    @abstractmethod
    def register_tools(self) -> Dict[str, Callable]:
        """Register and return tool functions."""
        pass
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get registered tool functions."""
        if not self.tool_functions and self.is_initialized:
            self.tool_functions = self.register_tools()
        return self.tool_functions


class ModuleRegistry:
    """Central registry for managing modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInterface] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.module_paths: Dict[str, str] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        
        # Module discovery paths
        self.discovery_paths = [
            "agents",
            "tools", 
            "modules",
            "extensions"
        ]
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_module(self, module: ModuleInterface, metadata: ModuleMetadata) -> bool:
        """Register a module with the registry."""
        with self._lock:
            try:
                module_id = metadata.id
                
                # Check for conflicts
                if module_id in self.modules:
                    existing_version = self.metadata[module_id].version
                    if self._compare_versions(metadata.version, existing_version) <= 0:
                        self.logger.warning(f"Module {module_id} version {metadata.version} "
                                          f"is not newer than existing {existing_version}")
                        return False
                
                # Set metadata
                module.metadata = metadata
                metadata.status = ModuleStatus.LOADED
                metadata.load_time = time.time()
                
                # Register
                self.modules[module_id] = module
                self.metadata[module_id] = metadata
                
                # Update dependency graph
                self._update_dependency_graph(module_id, metadata.dependencies)
                
                # Fire event
                self._fire_event("module_registered", module_id, metadata)
                
                logging.info(f"Module {module_id} v{metadata.version} registered successfully")
                return True
                
            except Exception as e:
                logging.error(f"Failed to register module {metadata.id}: {e}")
                metadata.status = ModuleStatus.ERROR
                metadata.error_message = str(e)
                return False
    
    def unregister_module(self, module_id: str) -> bool:
        """Unregister a module from the registry."""
        with self._lock:
            try:
                if module_id not in self.modules:
                    return False
                
                module = self.modules[module_id]
                metadata = self.metadata[module_id]
                
                # Check dependencies
                dependents = self._get_dependents(module_id)
                if dependents:
                    logging.warning(f"Cannot unregister {module_id}: has dependents {dependents}")
                    return False
                
                # Shutdown module
                if module.is_initialized:
                    module.shutdown()
                
                # Remove from registry
                del self.modules[module_id]
                del self.metadata[module_id]
                
                # Update dependency graph
                if module_id in self.dependency_graph:
                    del self.dependency_graph[module_id]
                
                # Fire event
                self._fire_event("module_unregistered", module_id, metadata)
                
                logging.info(f"Module {module_id} unregistered successfully")
                return True
                
            except Exception as e:
                logging.error(f"Failed to unregister module {module_id}: {e}")
                return False
    
    def get_module(self, module_id: str) -> Optional[ModuleInterface]:
        """Get a module by ID."""
        with self._lock:
            return self.modules.get(module_id)
    
    def get_metadata(self, module_id: str) -> Optional[ModuleMetadata]:
        """Get module metadata by ID."""
        with self._lock:
            return self.metadata.get(module_id)
    
    def list_modules(self, module_type: Optional[ModuleType] = None, 
                    status: Optional[ModuleStatus] = None) -> List[ModuleMetadata]:
        """List modules with optional filtering."""
        with self._lock:
            modules = list(self.metadata.values())
            
            if module_type:
                modules = [m for m in modules if m.module_type == module_type]
            
            if status:
                modules = [m for m in modules if m.status == status]
            
            return sorted(modules, key=lambda m: (m.module_type.value, m.name))
    
    def find_modules_by_capability(self, capability: str) -> List[ModuleMetadata]:
        """Find modules that provide a specific capability."""
        with self._lock:
            matching = []
            for metadata in self.metadata.values():
                if any(cap.name == capability for cap in metadata.capabilities):
                    matching.append(metadata)
            return matching
    
    def initialize_module(self, module_id: str) -> bool:
        """Initialize a specific module."""
        with self._lock:
            module = self.modules.get(module_id)
            if not module:
                return False
            
            try:
                # Check dependencies
                if not self._check_dependencies(module_id):
                    return False
                
                # Initialize
                metadata = self.metadata[module_id]
                metadata.status = ModuleStatus.LOADING
                
                success = module.initialize()
                
                if success:
                    module.is_initialized = True
                    metadata.status = ModuleStatus.ACTIVE
                    self._fire_event("module_initialized", module_id, metadata)
                    logging.info(f"Module {module_id} initialized successfully")
                else:
                    metadata.status = ModuleStatus.ERROR
                    metadata.error_message = "Initialization failed"
                    logging.error(f"Module {module_id} initialization failed")
                
                return success
                
            except Exception as e:
                metadata.status = ModuleStatus.ERROR
                metadata.error_message = str(e)
                logging.error(f"Module {module_id} initialization error: {e}")
                return False
    
    def shutdown_module(self, module_id: str) -> bool:
        """Shutdown a specific module."""
        with self._lock:
            module = self.modules.get(module_id)
            if not module:
                return False
            
            try:
                metadata = self.metadata[module_id]
                
                success = module.shutdown()
                
                if success:
                    module.is_initialized = False
                    metadata.status = ModuleStatus.LOADED
                    self._fire_event("module_shutdown", module_id, metadata)
                    logging.info(f"Module {module_id} shutdown successfully")
                else:
                    metadata.status = ModuleStatus.ERROR
                    metadata.error_message = "Shutdown failed"
                    logging.error(f"Module {module_id} shutdown failed")
                
                return success
                
            except Exception as e:
                metadata.status = ModuleStatus.ERROR
                metadata.error_message = str(e)
                logging.error(f"Module {module_id} shutdown error: {e}")
                return False
    
    def execute_capability(self, module_id: str, capability: str, **kwargs) -> Any:
        """Execute a capability on a specific module."""
        with self._lock:
            module = self.modules.get(module_id)
            if not module:
                raise ValueError(f"Module {module_id} not found")
            
            if not module.is_initialized:
                if not self.initialize_module(module_id):
                    raise RuntimeError(f"Failed to initialize module {module_id}")
            
            try:
                # Update usage statistics
                metadata = self.metadata[module_id]
                metadata.last_used = time.time()
                metadata.usage_count += 1
                
                # Execute capability
                result = module.execute(capability, **kwargs)
                
                self._fire_event("capability_executed", module_id, capability, result)
                return result
                
            except Exception as e:
                logging.error(f"Error executing {capability} on {module_id}: {e}")
                raise
    
    def add_event_handler(self, event: str, handler: Callable):
        """Add an event handler."""
        self.event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: str, handler: Callable):
        """Remove an event handler."""
        if handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)
    
    def _fire_event(self, event: str, *args, **kwargs):
        """Fire an event to all registered handlers."""
        for handler in self.event_handlers[event]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logging.error(f"Event handler error for {event}: {e}")
    
    def _check_dependencies(self, module_id: str) -> bool:
        """Check if module dependencies are satisfied."""
        metadata = self.metadata.get(module_id)
        if not metadata:
            return False
        
        for dep in metadata.dependencies:
            if dep.dependency_type == DependencyType.REQUIRED:
                if dep.name not in self.modules:
                    logging.error(f"Required dependency {dep.name} not found for {module_id}")
                    return False
                
                dep_module = self.modules[dep.name]
                if not dep_module.is_initialized:
                    if not self.initialize_module(dep.name):
                        logging.error(f"Failed to initialize dependency {dep.name} for {module_id}")
                        return False
        
        return True
    
    def _update_dependency_graph(self, module_id: str, dependencies: List[ModuleDependency]):
        """Update the dependency graph."""
        deps = [dep.name for dep in dependencies if dep.dependency_type == DependencyType.REQUIRED]
        self.dependency_graph[module_id] = deps
    
    def _get_dependents(self, module_id: str) -> List[str]:
        """Get modules that depend on the given module."""
        dependents = []
        for mod_id, deps in self.dependency_graph.items():
            if module_id in deps:
                dependents.append(mod_id)
        return dependents
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        v1 = version_tuple(version1)
        v2 = version_tuple(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    @property
    def logger(self):
        return logging.getLogger("module_registry")


class ModuleLoader:
    """Loads modules from various sources"""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.logger = logging.getLogger("module_loader")
    
    def load_from_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """Load modules from a directory."""
        loaded_modules = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return loaded_modules
        
        # Find module files
        pattern = "**/*.py" if recursive else "*.py"
        for module_file in directory_path.glob(pattern):
            if module_file.name.startswith("__"):
                continue
            
            try:
                module_id = self._load_module_file(module_file)
                if module_id:
                    loaded_modules.append(module_id)
            except Exception as e:
                self.logger.error(f"Failed to load module from {module_file}: {e}")
        
        return loaded_modules
    
    def load_from_manifest(self, manifest_path: str) -> List[str]:
        """Load modules from a manifest file."""
        loaded_modules = []
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            for module_info in manifest.get("modules", []):
                try:
                    module_id = self._load_from_manifest_entry(module_info)
                    if module_id:
                        loaded_modules.append(module_id)
                except Exception as e:
                    self.logger.error(f"Failed to load module from manifest entry: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to load manifest {manifest_path}: {e}")
        
        return loaded_modules
    
    def _load_module_file(self, module_file: Path) -> Optional[str]:
        """Load a module from a Python file."""
        try:
            # Add the module directory to Python path
            module_dir = str(module_file.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            
            # Import the module
            module_name = module_file.stem
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for module classes
            module_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ModuleInterface) and 
                    obj != ModuleInterface and
                    obj != AgentModule and
                    obj != ToolModule):
                    module_classes.append(obj)
            
            if not module_classes:
                self.logger.warning(f"No module classes found in {module_file}")
                return None
            
            # Use the first module class found
            module_class = module_classes[0]
            
            # Look for metadata
            metadata = self._extract_metadata(module, module_class, module_file)
            if not metadata:
                self.logger.warning(f"No metadata found for module in {module_file}")
                return None
            
            # Create module instance
            module_instance = module_class(metadata.id)
            
            # Register with registry
            if self.registry.register_module(module_instance, metadata):
                return metadata.id
            
        except Exception as e:
            self.logger.error(f"Error loading module from {module_file}: {e}")
            self.logger.debug(traceback.format_exc())
        
        return None
    
    def _extract_metadata(self, module, module_class, module_file: Path) -> Optional[ModuleMetadata]:
        """Extract metadata from a module."""
        try:
            # Look for metadata in various places
            metadata_dict = None
            
            # 1. Check for MODULE_METADATA constant
            if hasattr(module, 'MODULE_METADATA'):
                metadata_dict = module.MODULE_METADATA
            
            # 2. Check for get_metadata function
            elif hasattr(module, 'get_metadata'):
                metadata_dict = module.get_metadata()
            
            # 3. Check class docstring for metadata
            elif module_class.__doc__:
                metadata_dict = self._parse_docstring_metadata(module_class.__doc__)
            
            # 4. Generate basic metadata
            else:
                metadata_dict = {
                    "id": module_class.__name__.lower(),
                    "name": module_class.__name__,
                    "version": "1.0.0",
                    "description": module_class.__doc__ or "No description available"
                }
            
            # Ensure required fields
            if "id" not in metadata_dict:
                metadata_dict["id"] = module_class.__name__.lower()
            if "name" not in metadata_dict:
                metadata_dict["name"] = module_class.__name__
            if "version" not in metadata_dict:
                metadata_dict["version"] = "1.0.0"
            if "description" not in metadata_dict:
                metadata_dict["description"] = "No description available"
            
            # Determine module type
            if issubclass(module_class, AgentModule):
                module_type = ModuleType.AGENT
            elif issubclass(module_class, ToolModule):
                module_type = ModuleType.TOOL
            else:
                module_type = ModuleType.EXTENSION
            
            metadata_dict["module_type"] = module_type
            metadata_dict["module_path"] = str(module_file)
            
            # Convert to ModuleMetadata
            return self._dict_to_metadata(metadata_dict)
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return None
    
    def _dict_to_metadata(self, metadata_dict: Dict[str, Any]) -> ModuleMetadata:
        """Convert dictionary to ModuleMetadata object."""
        # Handle enum conversions
        if isinstance(metadata_dict.get("module_type"), str):
            metadata_dict["module_type"] = ModuleType(metadata_dict["module_type"])
        
        # Handle dependencies
        deps = []
        for dep_data in metadata_dict.get("dependencies", []):
            if isinstance(dep_data, dict):
                dep_type = DependencyType(dep_data.get("dependency_type", "required"))
                deps.append(ModuleDependency(
                    name=dep_data["name"],
                    version=dep_data.get("version"),
                    dependency_type=dep_type,
                    description=dep_data.get("description", "")
                ))
        metadata_dict["dependencies"] = deps
        
        # Handle capabilities
        caps = []
        for cap_data in metadata_dict.get("capabilities", []):
            if isinstance(cap_data, dict):
                caps.append(ModuleCapability(
                    name=cap_data["name"],
                    description=cap_data.get("description", ""),
                    input_schema=cap_data.get("input_schema", {}),
                    output_schema=cap_data.get("output_schema", {}),
                    parameters=cap_data.get("parameters", {}),
                    tags=cap_data.get("tags", [])
                ))
        metadata_dict["capabilities"] = caps
        
        # Create metadata object
        return ModuleMetadata(**metadata_dict)
    
    def _parse_docstring_metadata(self, docstring: str) -> Dict[str, Any]:
        """Parse metadata from docstring."""
        # Simple implementation - can be enhanced
        metadata = {}
        
        lines = docstring.strip().split('\n')
        if lines:
            metadata["description"] = lines[0].strip()
        
        # Look for metadata markers
        for line in lines:
            line = line.strip()
            if line.startswith("@version:"):
                metadata["version"] = line.split(":", 1)[1].strip()
            elif line.startswith("@author:"):
                metadata["author"] = line.split(":", 1)[1].strip()
            elif line.startswith("@id:"):
                metadata["id"] = line.split(":", 1)[1].strip()
        
        return metadata
    
    def _load_from_manifest_entry(self, module_info: Dict[str, Any]) -> Optional[str]:
        """Load a module from a manifest entry."""
        # Implementation for loading from manifest entries
        # This would handle different module sources (files, packages, URLs, etc.)
        pass


# Global module registry instance
_module_registry: Optional[ModuleRegistry] = None
_registry_lock = threading.Lock()


def get_module_registry() -> ModuleRegistry:
    """Get the global module registry instance."""
    global _module_registry
    
    with _registry_lock:
        if _module_registry is None:
            _module_registry = ModuleRegistry()
        return _module_registry


def initialize_module_system(discovery_paths: List[str] = None) -> ModuleRegistry:
    """Initialize the module system and discover modules."""
    registry = get_module_registry()
    loader = ModuleLoader(registry)
    
    # Set discovery paths
    if discovery_paths:
        registry.discovery_paths = discovery_paths
    
    # Load modules from discovery paths
    for path in registry.discovery_paths:
        if os.path.exists(path):
            loaded = loader.load_from_directory(path)
            logging.info(f"Loaded {len(loaded)} modules from {path}")
    
    return registry


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize module system
    registry = initialize_module_system()
    
    # List discovered modules
    modules = registry.list_modules()
    print(f"Discovered {len(modules)} modules:")
    
    for metadata in modules:
        print(f"  {metadata.name} v{metadata.version} ({metadata.module_type.value})")
        print(f"    {metadata.description}")
        if metadata.capabilities:
            print(f"    Capabilities: {[cap.name for cap in metadata.capabilities]}")
        print()