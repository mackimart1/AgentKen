"""
Setup script for the AgentKen Modular System
Initializes the modular framework and migrates existing agents and tools.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from module_system import (
    initialize_module_system, get_module_registry, ModuleType, ModuleStatus
)
from module_composer import get_composition_registry
from module_lifecycle import ModuleLifecycleManager


class ModularSystemSetup:
    """Setup and configuration for the modular system"""
    
    def __init__(self, config_path: str = "modular_config.json"):
        self.config_path = config_path
        self.config = self._load_or_create_config()
        self.registry = None
        self.lifecycle_manager = None
        self.composition_registry = None
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load configuration or create default"""
        default_config = {
            "system": {
                "discovery_paths": ["modules", "agents", "tools", "extensions"],
                "auto_initialize": True,
                "auto_migrate": True,
                "backup_existing": True
            },
            "modules": {
                "cache_dir": "module_cache",
                "temp_dir": "temp",
                "backup_dir": "module_backups",
                "max_file_size": 104857600,  # 100MB
                "supported_formats": ["py", "zip", "tar.gz"]
            },
            "lifecycle": {
                "auto_update": False,
                "update_check_interval": 86400,  # 24 hours
                "backup_before_update": True,
                "rollback_on_failure": True
            },
            "composition": {
                "enable_parallel": True,
                "max_parallel_steps": 10,
                "default_timeout": 300,  # 5 minutes
                "retry_failed_steps": True
            },
            "monitoring": {
                "health_check_interval": 300,  # 5 minutes
                "performance_tracking": True,
                "error_reporting": True,
                "usage_analytics": True
            },
            "security": {
                "validate_modules": True,
                "sandbox_execution": False,
                "allowed_imports": [],
                "restricted_operations": []
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
                logging.info("Using default configuration")
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def initialize_system(self) -> bool:
        """Initialize the modular system"""
        try:
            print("🚀 Initializing AgentKen Modular System...")
            
            # Create necessary directories
            self._create_directories()
            
            # Initialize module registry
            discovery_paths = self.config["system"]["discovery_paths"]
            self.registry = initialize_module_system(discovery_paths)
            
            # Initialize composition registry
            self.composition_registry = get_composition_registry()
            
            # Initialize lifecycle manager
            self.lifecycle_manager = ModuleLifecycleManager(self.registry)
            
            print("✅ Modular system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize modular system: {e}")
            logging.error(f"System initialization failed: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config["modules"]["cache_dir"],
            self.config["modules"]["temp_dir"],
            self.config["modules"]["backup_dir"],
            "modules",
            "extensions"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    def migrate_existing_components(self) -> Dict[str, Any]:
        """Migrate existing agents and tools to modular format"""
        if not self.config["system"]["auto_migrate"]:
            print("⚠️ Auto-migration disabled in configuration")
            return {"migrated": 0, "errors": []}
        
        print("🔄 Migrating existing components to modular format...")
        
        migration_results = {
            "migrated": 0,
            "errors": [],
            "agents": [],
            "tools": []
        }
        
        # Migrate agents
        agents_dir = Path("agents")
        if agents_dir.exists():
            migration_results["agents"] = self._migrate_directory(
                agents_dir, "agent", migration_results["errors"]
            )
            migration_results["migrated"] += len(migration_results["agents"])
        
        # Migrate tools
        tools_dir = Path("tools")
        if tools_dir.exists():
            migration_results["tools"] = self._migrate_directory(
                tools_dir, "tool", migration_results["errors"]
            )
            migration_results["migrated"] += len(migration_results["tools"])
        
        print(f"✅ Migration completed: {migration_results['migrated']} components migrated")
        if migration_results["errors"]:
            print(f"⚠️ {len(migration_results['errors'])} errors occurred during migration")
        
        return migration_results
    
    def _migrate_directory(self, source_dir: Path, component_type: str, errors: List[str]) -> List[str]:
        """Migrate components from a directory"""
        migrated = []
        
        for py_file in source_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                # Check if already modular
                if self._is_modular_component(py_file):
                    logging.info(f"{py_file.name} is already modular")
                    continue
                
                # Create migration wrapper
                wrapper_content = self._create_migration_wrapper(py_file, component_type)
                
                # Save to modules directory
                modules_dir = Path("modules")
                target_file = modules_dir / f"migrated_{py_file.name}"
                
                with open(target_file, 'w') as f:
                    f.write(wrapper_content)
                
                migrated.append(py_file.name)
                logging.info(f"Migrated {py_file.name} to modular format")
                
            except Exception as e:
                error_msg = f"Failed to migrate {py_file.name}: {e}"
                errors.append(error_msg)
                logging.error(error_msg)
        
        return migrated
    
    def _is_modular_component(self, file_path: Path) -> bool:
        """Check if a component is already modular"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for modular indicators
            modular_indicators = [
                "ModuleInterface",
                "AgentModule", 
                "ToolModule",
                "MODULE_METADATA",
                "get_metadata"
            ]
            
            return any(indicator in content for indicator in modular_indicators)
            
        except Exception:
            return False
    
    def _create_migration_wrapper(self, original_file: Path, component_type: str) -> str:
        """Create a migration wrapper for existing components"""
        component_name = original_file.stem
        class_name = f"Migrated{component_name.title().replace('_', '')}"
        
        wrapper_template = f'''"""
Migrated {component_type} module: {component_name}
Auto-generated wrapper for existing {component_type}.
"""

import sys
import os
import logging
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '{original_file.parent.name}'))

from module_system import {"AgentModule" if component_type == "agent" else "ToolModule"}, ModuleCapability, ModuleMetadata, ModuleType

# Import original component
try:
    from {original_file.stem} import *
except ImportError as e:
    logging.error(f"Failed to import original {component_type}: {{e}}")


class {class_name}({"AgentModule" if component_type == "agent" else "ToolModule"}):
    """Migrated {component_type} wrapper"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        super().__init__(module_id, config)
        
        # Initialize metadata
        self.metadata = ModuleMetadata(
            id="{component_name}_migrated",
            name="{component_name.title().replace('_', ' ')}",
            version="1.0.0",
            module_type=ModuleType.{"AGENT" if component_type == "agent" else "TOOL"},
            description="Migrated {component_type} from existing codebase",
            author="AgentKen Migration",
            
            capabilities=[
                ModuleCapability(
                    name="execute",
                    description="Execute the migrated {component_type}",
                    input_schema={{"type": "object"}},
                    output_schema={{"type": "object"}},
                    tags=["migrated", "{component_type}"]
                )
            ],
            
            tags=["migrated", "{component_type}", "{component_name}"],
            category="migrated"
        )
    
    def initialize(self) -> bool:
        """Initialize the migrated {component_type}"""
        try:
            # Initialize original component if needed
            self.logger.info("Migrated {component_type} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize migrated {component_type}: {{e}}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the migrated {component_type}"""
        try:
            self.logger.info("Migrated {component_type} shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown migrated {component_type}: {{e}}")
            return False
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """Return list of capabilities"""
        return self.metadata.capabilities
    
    def execute(self, capability: str, **kwargs) -> Any:
        """Execute a capability"""
        if capability == "execute":
            # Call original component logic here
            return self._execute_original(**kwargs)
        else:
            raise ValueError(f"Unknown capability: {{capability}}")
    
    def _execute_original(self, **kwargs) -> Any:
        """Execute original component logic"""
        # This would need to be customized based on the original component
        return {{"status": "executed", "component": "{component_name}"}}

{"" if component_type == "tool" else '''
    def create_agent(self) -> Any:
        """Create and return the agent instance"""
        # Return original agent instance or wrapper
        return self
'''}

{"" if component_type == "agent" else '''
    def register_tools(self) -> Dict[str, callable]:
        """Register and return tool functions"""
        return {{
            "execute": self._execute_original
        }}
'''}


# Module metadata for discovery
MODULE_METADATA = {{
    "id": "{component_name}_migrated",
    "name": "{component_name.title().replace('_', ' ')}",
    "version": "1.0.0",
    "module_type": "{component_type}",
    "description": "Migrated {component_type} from existing codebase",
    "author": "AgentKen Migration",
    "capabilities": [
        {{
            "name": "execute",
            "description": "Execute the migrated {component_type}",
            "tags": ["migrated", "{component_type}"]
        }}
    ],
    "tags": ["migrated", "{component_type}", "{component_name}"]
}}


def get_metadata():
    """Function to get module metadata"""
    return MODULE_METADATA
'''
        
        return wrapper_template
    
    def discover_and_load_modules(self) -> Dict[str, Any]:
        """Discover and load all available modules"""
        print("🔍 Discovering and loading modules...")
        
        if not self.registry:
            print("❌ Registry not initialized")
            return {"loaded": 0, "errors": []}
        
        # Get discovered modules
        modules = self.registry.list_modules()
        
        results = {
            "discovered": len(modules),
            "loaded": 0,
            "initialized": 0,
            "errors": []
        }
        
        print(f"📦 Discovered {results['discovered']} modules")
        
        # Initialize modules if configured
        if self.config["system"]["auto_initialize"]:
            print("🚀 Auto-initializing modules...")
            
            for metadata in modules:
                if metadata.status == ModuleStatus.LOADED:
                    try:
                        success = self.registry.initialize_module(metadata.id)
                        if success:
                            results["initialized"] += 1
                            print(f"  ✅ {metadata.name} initialized")
                        else:
                            error_msg = f"Failed to initialize {metadata.name}"
                            results["errors"].append(error_msg)
                            print(f"  ❌ {error_msg}")
                    except Exception as e:
                        error_msg = f"Error initializing {metadata.name}: {e}"
                        results["errors"].append(error_msg)
                        print(f"  ❌ {error_msg}")
        
        print(f"✅ Module loading completed: {results['initialized']} initialized")
        return results
    
    def create_example_compositions(self) -> List[str]:
        """Create example compositions to demonstrate the system"""
        print("🔧 Creating example compositions...")
        
        if not self.composition_registry:
            print("❌ Composition registry not initialized")
            return []
        
        created_compositions = []
        
        # Find available modules
        research_modules = self.registry.find_modules_by_capability("web_search")
        data_modules = self.registry.find_modules_by_capability("clean_data")
        
        try:
            # Create research pipeline if modules available
            if research_modules:
                from module_composer import CompositionBuilder, CompositionType, ExecutionMode
                
                research_module_id = research_modules[0].id
                
                builder = CompositionBuilder(
                    "example_research_pipeline",
                    "Example Research Pipeline",
                    CompositionType.PIPELINE
                )
                
                composition = (builder
                              .set_description("Example pipeline for web research and analysis")
                              .set_execution_mode(ExecutionMode.SYNCHRONOUS)
                              .add_step(
                                  "search_step",
                                  research_module_id,
                                  "web_search",
                                  parameters={"max_results": 5},
                                  input_mapping={"query": "search_query"}
                              )
                              .build())
                
                if self.composition_registry.register_composition(composition):
                    created_compositions.append("example_research_pipeline")
                    print("  ✅ Research pipeline composition created")
            
            # Create data processing pipeline if modules available
            if data_modules:
                data_module_id = data_modules[0].id
                
                builder = CompositionBuilder(
                    "example_data_pipeline",
                    "Example Data Processing Pipeline", 
                    CompositionType.PIPELINE
                )
                
                composition = (builder
                              .set_description("Example pipeline for data cleaning and analysis")
                              .add_step(
                                  "clean_step",
                                  data_module_id,
                                  "clean_data",
                                  parameters={"operations": ["remove_duplicates", "handle_missing"]},
                                  input_mapping={"data": "input_data"}
                              )
                              .add_step(
                                  "analyze_step",
                                  data_module_id,
                                  "analyze_data",
                                  parameters={"analysis_type": "descriptive"},
                                  input_mapping={"data": "cleaned_data"}
                              )
                              .build())
                
                if self.composition_registry.register_composition(composition):
                    created_compositions.append("example_data_pipeline")
                    print("  ✅ Data processing pipeline composition created")
        
        except Exception as e:
            print(f"  ❌ Failed to create example compositions: {e}")
        
        print(f"✅ Created {len(created_compositions)} example compositions")
        return created_compositions
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report"""
        if not self.registry:
            return {"error": "System not initialized"}
        
        # Module statistics
        all_modules = self.registry.list_modules()
        loaded_modules = self.registry.list_modules(status=ModuleStatus.LOADED)
        active_modules = self.registry.list_modules(status=ModuleStatus.ACTIVE)
        
        # Module types
        agents = self.registry.list_modules(module_type=ModuleType.AGENT)
        tools = self.registry.list_modules(module_type=ModuleType.TOOL)
        
        # Compositions
        compositions = self.composition_registry.list_compositions() if self.composition_registry else []
        
        # Capabilities
        all_capabilities = set()
        for metadata in all_modules:
            for capability in metadata.capabilities:
                all_capabilities.add(capability.name)
        
        return {
            "system_status": "operational",
            "modules": {
                "total": len(all_modules),
                "loaded": len(loaded_modules),
                "active": len(active_modules),
                "agents": len(agents),
                "tools": len(tools)
            },
            "compositions": {
                "total": len(compositions)
            },
            "capabilities": {
                "total": len(all_capabilities),
                "list": sorted(all_capabilities)
            },
            "configuration": self.config,
            "directories_created": [
                self.config["modules"]["cache_dir"],
                self.config["modules"]["temp_dir"],
                self.config["modules"]["backup_dir"]
            ]
        }


def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 AgentKen Modular System Setup")
    print("=" * 50)
    
    # Initialize setup
    setup = ModularSystemSetup()
    
    # Initialize system
    if not setup.initialize_system():
        print("❌ System initialization failed")
        return
    
    # Migrate existing components
    migration_results = setup.migrate_existing_components()
    
    # Discover and load modules
    loading_results = setup.discover_and_load_modules()
    
    # Create example compositions
    example_compositions = setup.create_example_compositions()
    
    # Generate setup report
    report = setup.generate_setup_report()
    
    print("\n📋 Setup Report:")
    print(f"  System Status: {report['system_status']}")
    print(f"  Total Modules: {report['modules']['total']}")
    print(f"  Active Modules: {report['modules']['active']}")
    print(f"  Agents: {report['modules']['agents']}")
    print(f"  Tools: {report['modules']['tools']}")
    print(f"  Compositions: {report['compositions']['total']}")
    print(f"  Capabilities: {report['capabilities']['total']}")
    print(f"  Migrated Components: {migration_results['migrated']}")
    
    print("\n🎉 Modular system setup complete!")
    print("\nNext steps:")
    print("1. Run the test suite: python test_modular_system.py")
    print("2. Create custom modules in the modules/ directory")
    print("3. Build compositions using the composition framework")
    print("4. Monitor system health and performance")
    print("5. Use the lifecycle manager for updates and maintenance")
    
    # Save final configuration
    setup._save_config(setup.config)
    
    print(f"\nConfiguration saved to: {setup.config_path}")
    print("System is ready for use!")


if __name__ == "__main__":
    main()