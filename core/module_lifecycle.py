"""
Module Lifecycle Management
Handles module upgrades, versioning, hot-swapping, and lifecycle events.
"""

import json
import logging
import os
import shutil
import time
import threading
import zipfile
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import requests
import hashlib
import tempfile

from module_system import (
    ModuleRegistry, ModuleInterface, ModuleMetadata, ModuleStatus, ModuleType,
    get_module_registry
)


class UpgradeType(Enum):
    """Types of module upgrades"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    HOTFIX = "hotfix"


class LifecycleEvent(Enum):
    """Module lifecycle events"""
    INSTALLED = "installed"
    LOADED = "loaded"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    UPGRADED = "upgraded"
    DOWNGRADED = "downgraded"
    UNINSTALLED = "uninstalled"
    ERROR = "error"


@dataclass
class ModuleVersion:
    """Represents a module version"""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ModuleVersion':
        """Create ModuleVersion from string."""
        parts = version_str.split('.')
        if len(parts) < 3:
            raise ValueError(f"Invalid version string: {version_str}")
        
        major, minor, patch = map(int, parts[:3])
        prerelease = parts[3] if len(parts) > 3 else ""
        build = parts[4] if len(parts) > 4 else ""
        
        return cls(major, minor, patch, prerelease, build)
    
    def to_string(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f".{self.prerelease}"
        if self.build:
            version += f".{self.build}"
        return version
    
    def compare(self, other: 'ModuleVersion') -> int:
        """Compare with another version. Returns -1, 0, or 1."""
        if self.major != other.major:
            return 1 if self.major > other.major else -1
        if self.minor != other.minor:
            return 1 if self.minor > other.minor else -1
        if self.patch != other.patch:
            return 1 if self.patch > other.patch else -1
        
        # Handle prerelease comparison
        if self.prerelease and not other.prerelease:
            return -1
        if not self.prerelease and other.prerelease:
            return 1
        if self.prerelease and other.prerelease:
            return 1 if self.prerelease > other.prerelease else (-1 if self.prerelease < other.prerelease else 0)
        
        return 0
    
    def is_compatible(self, other: 'ModuleVersion') -> bool:
        """Check if this version is compatible with another."""
        # Same major version is considered compatible
        return self.major == other.major


@dataclass
class ModulePackage:
    """Represents a module package"""
    id: str
    name: str
    version: ModuleVersion
    description: str
    author: str
    license: str
    homepage: str
    download_url: str
    checksum: str
    size: int
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class LifecycleEventRecord:
    """Record of a lifecycle event"""
    module_id: str
    event: LifecycleEvent
    timestamp: float
    version: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ModuleRepository:
    """Manages module packages and repositories"""
    
    def __init__(self, repository_url: str = None):
        self.repository_url = repository_url or "https://agentken.io/modules"
        self.cache_dir = Path("module_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("module_repository")
    
    def search_modules(self, query: str = "", module_type: Optional[ModuleType] = None,
                      tags: List[str] = None) -> List[ModulePackage]:
        """Search for modules in the repository."""
        try:
            params = {"q": query}
            if module_type:
                params["type"] = module_type.value
            if tags:
                params["tags"] = ",".join(tags)
            
            response = requests.get(f"{self.repository_url}/search", params=params)
            response.raise_for_status()
            
            packages = []
            for package_data in response.json().get("packages", []):
                package = self._dict_to_package(package_data)
                packages.append(package)
            
            return packages
            
        except Exception as e:
            self.logger.error(f"Failed to search modules: {e}")
            return []
    
    def get_module_info(self, module_id: str, version: str = "latest") -> Optional[ModulePackage]:
        """Get information about a specific module."""
        try:
            url = f"{self.repository_url}/modules/{module_id}"
            if version != "latest":
                url += f"/{version}"
            
            response = requests.get(url)
            response.raise_for_status()
            
            package_data = response.json()
            return self._dict_to_package(package_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get module info for {module_id}: {e}")
            return None
    
    def download_module(self, package: ModulePackage, target_dir: Path) -> bool:
        """Download a module package."""
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                response = requests.get(package.download_url, stream=True)
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"Downloading {package.name}: {progress:.1f}%")
                
                temp_file_path = temp_file.name
            
            # Verify checksum
            if not self._verify_checksum(temp_file_path, package.checksum):
                os.unlink(temp_file_path)
                self.logger.error(f"Checksum verification failed for {package.name}")
                return False
            
            # Extract to target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(temp_file_path, 'r') as zip_file:
                zip_file.extractall(target_dir)
            
            # Clean up
            os.unlink(temp_file_path)
            
            self.logger.info(f"Successfully downloaded {package.name} v{package.version.to_string()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {package.name}: {e}")
            return False
    
    def _dict_to_package(self, package_data: Dict[str, Any]) -> ModulePackage:
        """Convert dictionary to ModulePackage."""
        version = ModuleVersion.from_string(package_data["version"])
        
        return ModulePackage(
            id=package_data["id"],
            name=package_data["name"],
            version=version,
            description=package_data.get("description", ""),
            author=package_data.get("author", ""),
            license=package_data.get("license", ""),
            homepage=package_data.get("homepage", ""),
            download_url=package_data["download_url"],
            checksum=package_data["checksum"],
            size=package_data.get("size", 0),
            dependencies=package_data.get("dependencies", []),
            tags=package_data.get("tags", []),
            created_at=package_data.get("created_at", time.time()),
            updated_at=package_data.get("updated_at", time.time())
        )
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Verify file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest() == expected_checksum


class ModuleLifecycleManager:
    """Manages module lifecycle operations"""
    
    def __init__(self, registry: ModuleRegistry = None, repository: ModuleRepository = None):
        self.registry = registry or get_module_registry()
        self.repository = repository or ModuleRepository()
        self.modules_dir = Path("modules")
        self.modules_dir.mkdir(exist_ok=True)
        self.backup_dir = Path("module_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Event tracking
        self.event_history: List[LifecycleEventRecord] = []
        self.event_handlers: Dict[LifecycleEvent, List[Callable]] = {}
        
        # Lifecycle state
        self.installed_modules: Dict[str, ModulePackage] = {}
        self.upgrade_queue: List[str] = []
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger("module_lifecycle")
        
        # Load installed modules
        self._load_installed_modules()
    
    def install_module(self, module_id: str, version: str = "latest", 
                      force: bool = False) -> bool:
        """Install a module from the repository."""
        with self._lock:
            try:
                # Check if already installed
                if module_id in self.installed_modules and not force:
                    self.logger.info(f"Module {module_id} is already installed")
                    return True
                
                # Get module info
                package = self.repository.get_module_info(module_id, version)
                if not package:
                    self.logger.error(f"Module {module_id} not found in repository")
                    return False
                
                # Check dependencies
                if not self._check_dependencies(package):
                    self.logger.error(f"Dependency check failed for {module_id}")
                    return False
                
                # Download module
                module_dir = self.modules_dir / module_id
                if module_dir.exists() and not force:
                    shutil.rmtree(module_dir)
                
                if not self.repository.download_module(package, module_dir):
                    return False
                
                # Load module
                from module_system import ModuleLoader
                loader = ModuleLoader(self.registry)
                loaded_modules = loader.load_from_directory(str(module_dir))
                
                if not loaded_modules:
                    self.logger.error(f"Failed to load module {module_id}")
                    return False
                
                # Update installed modules
                self.installed_modules[module_id] = package
                self._save_installed_modules()
                
                # Record event
                self._record_event(module_id, LifecycleEvent.INSTALLED, package.version.to_string())
                
                self.logger.info(f"Successfully installed {module_id} v{package.version.to_string()}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to install module {module_id}: {e}")
                self._record_event(module_id, LifecycleEvent.ERROR, "", error=str(e))
                return False
    
    def uninstall_module(self, module_id: str, force: bool = False) -> bool:
        """Uninstall a module."""
        with self._lock:
            try:
                # Check if installed
                if module_id not in self.installed_modules:
                    self.logger.warning(f"Module {module_id} is not installed")
                    return True
                
                # Check dependencies
                if not force:
                    dependents = self._get_dependent_modules(module_id)
                    if dependents:
                        self.logger.error(f"Cannot uninstall {module_id}: required by {dependents}")
                        return False
                
                # Unregister from registry
                self.registry.unregister_module(module_id)
                
                # Remove module files
                module_dir = self.modules_dir / module_id
                if module_dir.exists():
                    # Create backup before removal
                    backup_path = self.backup_dir / f"{module_id}_{int(time.time())}"
                    shutil.move(str(module_dir), str(backup_path))
                
                # Update installed modules
                package = self.installed_modules[module_id]
                del self.installed_modules[module_id]
                self._save_installed_modules()
                
                # Record event
                self._record_event(module_id, LifecycleEvent.UNINSTALLED, package.version.to_string())
                
                self.logger.info(f"Successfully uninstalled {module_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to uninstall module {module_id}: {e}")
                self._record_event(module_id, LifecycleEvent.ERROR, "", error=str(e))
                return False
    
    def upgrade_module(self, module_id: str, target_version: str = "latest",
                      backup: bool = True) -> bool:
        """Upgrade a module to a newer version."""
        with self._lock:
            try:
                # Check if installed
                if module_id not in self.installed_modules:
                    self.logger.error(f"Module {module_id} is not installed")
                    return False
                
                current_package = self.installed_modules[module_id]
                current_version = current_package.version
                
                # Get target package info
                target_package = self.repository.get_module_info(module_id, target_version)
                if not target_package:
                    self.logger.error(f"Target version {target_version} not found for {module_id}")
                    return False
                
                target_version_obj = target_package.version
                
                # Check if upgrade is needed
                if current_version.compare(target_version_obj) >= 0:
                    self.logger.info(f"Module {module_id} is already at version {current_version.to_string()} or newer")
                    return True
                
                # Determine upgrade type
                upgrade_type = self._determine_upgrade_type(current_version, target_version_obj)
                
                # Create backup if requested
                backup_path = None
                if backup:
                    backup_path = self._create_module_backup(module_id)
                
                # Perform hot-swap if possible
                if self._can_hot_swap(module_id, upgrade_type):
                    success = self._perform_hot_swap(module_id, target_package)
                else:
                    success = self._perform_cold_upgrade(module_id, target_package)
                
                if success:
                    # Update installed modules
                    self.installed_modules[module_id] = target_package
                    self._save_installed_modules()
                    
                    # Record event
                    self._record_event(
                        module_id, LifecycleEvent.UPGRADED, 
                        target_version_obj.to_string(),
                        details={
                            "from_version": current_version.to_string(),
                            "upgrade_type": upgrade_type.value,
                            "backup_path": str(backup_path) if backup_path else None
                        }
                    )
                    
                    self.logger.info(f"Successfully upgraded {module_id} from {current_version.to_string()} to {target_version_obj.to_string()}")
                else:
                    # Restore backup if upgrade failed
                    if backup_path:
                        self._restore_module_backup(module_id, backup_path)
                
                return success
                
            except Exception as e:
                self.logger.error(f"Failed to upgrade module {module_id}: {e}")
                self._record_event(module_id, LifecycleEvent.ERROR, "", error=str(e))
                return False
    
    def check_for_updates(self) -> Dict[str, ModulePackage]:
        """Check for available updates for installed modules."""
        updates = {}
        
        for module_id, current_package in self.installed_modules.items():
            try:
                latest_package = self.repository.get_module_info(module_id, "latest")
                if latest_package and latest_package.version.compare(current_package.version) > 0:
                    updates[module_id] = latest_package
            except Exception as e:
                self.logger.error(f"Failed to check updates for {module_id}: {e}")
        
        return updates
    
    def auto_update_modules(self, update_type: UpgradeType = UpgradeType.PATCH) -> Dict[str, bool]:
        """Automatically update modules based on update type."""
        results = {}
        updates = self.check_for_updates()
        
        for module_id, target_package in updates.items():
            current_package = self.installed_modules[module_id]
            upgrade_type = self._determine_upgrade_type(current_package.version, target_package.version)
            
            # Only update if upgrade type is allowed
            if self._is_upgrade_allowed(upgrade_type, update_type):
                results[module_id] = self.upgrade_module(module_id, target_package.version.to_string())
            else:
                self.logger.info(f"Skipping {module_id} upgrade: {upgrade_type.value} not allowed")
                results[module_id] = False
        
        return results
    
    def rollback_module(self, module_id: str, target_version: str) -> bool:
        """Rollback a module to a previous version."""
        with self._lock:
            try:
                # Check if installed
                if module_id not in self.installed_modules:
                    self.logger.error(f"Module {module_id} is not installed")
                    return False
                
                current_package = self.installed_modules[module_id]
                current_version = current_package.version
                
                # Get target package info
                target_package = self.repository.get_module_info(module_id, target_version)
                if not target_package:
                    self.logger.error(f"Target version {target_version} not found for {module_id}")
                    return False
                
                target_version_obj = target_package.version
                
                # Check if rollback is valid
                if current_version.compare(target_version_obj) <= 0:
                    self.logger.error(f"Cannot rollback {module_id} to same or newer version")
                    return False
                
                # Perform rollback (similar to upgrade but in reverse)
                success = self._perform_cold_upgrade(module_id, target_package)
                
                if success:
                    # Update installed modules
                    self.installed_modules[module_id] = target_package
                    self._save_installed_modules()
                    
                    # Record event
                    self._record_event(
                        module_id, LifecycleEvent.DOWNGRADED,
                        target_version_obj.to_string(),
                        details={
                            "from_version": current_version.to_string()
                        }
                    )
                    
                    self.logger.info(f"Successfully rolled back {module_id} from {current_version.to_string()} to {target_version_obj.to_string()}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Failed to rollback module {module_id}: {e}")
                self._record_event(module_id, LifecycleEvent.ERROR, "", error=str(e))
                return False
    
    def get_module_status(self, module_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a module."""
        status = {
            "installed": module_id in self.installed_modules,
            "loaded": module_id in self.registry.modules,
            "active": False,
            "version": None,
            "latest_version": None,
            "update_available": False,
            "dependencies_satisfied": False,
            "last_event": None
        }
        
        if module_id in self.installed_modules:
            package = self.installed_modules[module_id]
            status["version"] = package.version.to_string()
            
            # Check for updates
            latest_package = self.repository.get_module_info(module_id, "latest")
            if latest_package:
                status["latest_version"] = latest_package.version.to_string()
                status["update_available"] = latest_package.version.compare(package.version) > 0
            
            # Check dependencies
            status["dependencies_satisfied"] = self._check_dependencies(package)
        
        if module_id in self.registry.modules:
            module = self.registry.modules[module_id]
            status["active"] = module.is_initialized
        
        # Get last event
        for event in reversed(self.event_history):
            if event.module_id == module_id:
                status["last_event"] = {
                    "event": event.event.value,
                    "timestamp": event.timestamp,
                    "version": event.version,
                    "error": event.error
                }
                break
        
        return status
    
    def list_installed_modules(self) -> List[Dict[str, Any]]:
        """List all installed modules with their status."""
        modules = []
        
        for module_id, package in self.installed_modules.items():
            status = self.get_module_status(module_id)
            modules.append({
                "id": module_id,
                "name": package.name,
                "version": package.version.to_string(),
                "description": package.description,
                "author": package.author,
                "status": status
            })
        
        return sorted(modules, key=lambda m: m["name"])
    
    def add_event_handler(self, event: LifecycleEvent, handler: Callable):
        """Add an event handler for lifecycle events."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def _check_dependencies(self, package: ModulePackage) -> bool:
        """Check if package dependencies are satisfied."""
        for dep_id in package.dependencies:
            if dep_id not in self.installed_modules:
                return False
        return True
    
    def _get_dependent_modules(self, module_id: str) -> List[str]:
        """Get modules that depend on the given module."""
        dependents = []
        
        for other_id, other_package in self.installed_modules.items():
            if module_id in other_package.dependencies:
                dependents.append(other_id)
        
        return dependents
    
    def _determine_upgrade_type(self, current: ModuleVersion, target: ModuleVersion) -> UpgradeType:
        """Determine the type of upgrade."""
        if target.major > current.major:
            return UpgradeType.MAJOR
        elif target.minor > current.minor:
            return UpgradeType.MINOR
        elif target.patch > current.patch:
            return UpgradeType.PATCH
        else:
            return UpgradeType.HOTFIX
    
    def _is_upgrade_allowed(self, upgrade_type: UpgradeType, allowed_type: UpgradeType) -> bool:
        """Check if upgrade type is allowed."""
        type_hierarchy = [UpgradeType.HOTFIX, UpgradeType.PATCH, UpgradeType.MINOR, UpgradeType.MAJOR]
        return type_hierarchy.index(upgrade_type) <= type_hierarchy.index(allowed_type)
    
    def _can_hot_swap(self, module_id: str, upgrade_type: UpgradeType) -> bool:
        """Check if module can be hot-swapped."""
        # Hot-swap is only safe for patch and hotfix upgrades
        if upgrade_type not in [UpgradeType.PATCH, UpgradeType.HOTFIX]:
            return False
        
        # Check if module is currently in use
        module = self.registry.get_module(module_id)
        if not module or not module.is_initialized:
            return True
        
        # Additional checks can be added here
        return True
    
    def _perform_hot_swap(self, module_id: str, target_package: ModulePackage) -> bool:
        """Perform hot-swap upgrade."""
        try:
            # Download new version to temporary location
            temp_dir = Path(tempfile.mkdtemp())
            
            if not self.repository.download_module(target_package, temp_dir):
                shutil.rmtree(temp_dir)
                return False
            
            # Load new module version
            from module_system import ModuleLoader
            temp_registry = ModuleRegistry()
            loader = ModuleLoader(temp_registry)
            loaded_modules = loader.load_from_directory(str(temp_dir))
            
            if not loaded_modules:
                shutil.rmtree(temp_dir)
                return False
            
            # Get new module instance
            new_module = temp_registry.get_module(module_id)
            if not new_module:
                shutil.rmtree(temp_dir)
                return False
            
            # Swap modules
            old_module = self.registry.get_module(module_id)
            if old_module and old_module.is_initialized:
                old_module.shutdown()
            
            # Replace in registry
            self.registry.modules[module_id] = new_module
            self.registry.metadata[module_id] = new_module.metadata
            
            # Initialize new module
            if not new_module.initialize():
                # Rollback on failure
                self.registry.modules[module_id] = old_module
                if old_module:
                    old_module.initialize()
                shutil.rmtree(temp_dir)
                return False
            
            # Replace module files
            module_dir = self.modules_dir / module_id
            backup_dir = self.backup_dir / f"{module_id}_hotswap_{int(time.time())}"
            shutil.move(str(module_dir), str(backup_dir))
            shutil.move(str(temp_dir), str(module_dir))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hot-swap failed for {module_id}: {e}")
            return False
    
    def _perform_cold_upgrade(self, module_id: str, target_package: ModulePackage) -> bool:
        """Perform cold upgrade (requires restart)."""
        try:
            # Shutdown current module
            module = self.registry.get_module(module_id)
            if module and module.is_initialized:
                module.shutdown()
            
            # Unregister current module
            self.registry.unregister_module(module_id)
            
            # Download new version
            module_dir = self.modules_dir / module_id
            temp_dir = Path(tempfile.mkdtemp())
            
            if not self.repository.download_module(target_package, temp_dir):
                shutil.rmtree(temp_dir)
                return False
            
            # Replace module files
            if module_dir.exists():
                shutil.rmtree(module_dir)
            shutil.move(str(temp_dir), str(module_dir))
            
            # Load new module
            from module_system import ModuleLoader
            loader = ModuleLoader(self.registry)
            loaded_modules = loader.load_from_directory(str(module_dir))
            
            if not loaded_modules:
                return False
            
            # Initialize new module
            return self.registry.initialize_module(module_id)
            
        except Exception as e:
            self.logger.error(f"Cold upgrade failed for {module_id}: {e}")
            return False
    
    def _create_module_backup(self, module_id: str) -> Optional[Path]:
        """Create a backup of a module."""
        try:
            module_dir = self.modules_dir / module_id
            if not module_dir.exists():
                return None
            
            backup_path = self.backup_dir / f"{module_id}_{int(time.time())}"
            shutil.copytree(str(module_dir), str(backup_path))
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {module_id}: {e}")
            return None
    
    def _restore_module_backup(self, module_id: str, backup_path: Path) -> bool:
        """Restore a module from backup."""
        try:
            module_dir = self.modules_dir / module_id
            
            if module_dir.exists():
                shutil.rmtree(module_dir)
            
            shutil.copytree(str(backup_path), str(module_dir))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup for {module_id}: {e}")
            return False
    
    def _record_event(self, module_id: str, event: LifecycleEvent, version: str,
                     details: Dict[str, Any] = None, error: str = None):
        """Record a lifecycle event."""
        event_record = LifecycleEventRecord(
            module_id=module_id,
            event=event,
            timestamp=time.time(),
            version=version,
            details=details or {},
            error=error
        )
        
        self.event_history.append(event_record)
        
        # Fire event handlers
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(event_record)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    def _load_installed_modules(self):
        """Load installed modules from disk."""
        installed_file = self.modules_dir / "installed.json"
        
        if installed_file.exists():
            try:
                with open(installed_file, 'r') as f:
                    data = json.load(f)
                
                for module_id, package_data in data.items():
                    version = ModuleVersion.from_string(package_data["version"])
                    package_data["version"] = version
                    
                    package = ModulePackage(**package_data)
                    self.installed_modules[module_id] = package
                    
            except Exception as e:
                self.logger.error(f"Failed to load installed modules: {e}")
    
    def _save_installed_modules(self):
        """Save installed modules to disk."""
        installed_file = self.modules_dir / "installed.json"
        
        try:
            data = {}
            for module_id, package in self.installed_modules.items():
                package_data = asdict(package)
                package_data["version"] = package.version.to_string()
                data[module_id] = package_data
            
            with open(installed_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save installed modules: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize lifecycle manager
    lifecycle_manager = ModuleLifecycleManager()
    
    # Check for updates
    updates = lifecycle_manager.check_for_updates()
    print(f"Available updates: {len(updates)}")
    
    # List installed modules
    installed = lifecycle_manager.list_installed_modules()
    print(f"Installed modules: {len(installed)}")
    
    for module in installed:
        print(f"  {module['name']} v{module['version']}")
    
    print("Module lifecycle system initialized successfully")