"""
Deployment and Configuration Management System
Provides comprehensive deployment, configuration, and environment management capabilities.
"""

import os
import json
import yaml
import logging
import time
import shutil
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import threading
from collections import defaultdict
import hashlib


class DeploymentStatus(Enum):
    """Deployment status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class EnvironmentType(Enum):
    """Environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Configuration file formats"""

    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"


@dataclass
class DeploymentTarget:
    """Deployment target configuration"""

    name: str
    type: str  # local, docker, kubernetes, cloud
    environment: EnvironmentType
    connection_params: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ConfigurationItem:
    """Individual configuration item"""

    key: str
    value: Any
    description: str = ""
    required: bool = True
    sensitive: bool = False
    validation_rules: List[str] = field(default_factory=list)
    environment_specific: bool = False


@dataclass
class DeploymentPackage:
    """Deployment package information"""

    name: str
    version: str
    description: str
    artifacts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    configurations: Dict[str, ConfigurationItem] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)  # lifecycle scripts
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Record of a deployment"""

    deployment_id: str
    package_name: str
    package_version: str
    target_name: str
    status: DeploymentStatus
    start_time: float
    end_time: Optional[float] = None
    deployed_by: str = "system"
    configuration_hash: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None
    logs: List[str] = field(default_factory=list)
    artifacts_path: Optional[str] = None


class ConfigurationManager:
    """Manages configuration files and environment-specific settings"""

    def __init__(self, config_directory: str = "configs"):
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)

        self.configurations: Dict[str, Dict[str, ConfigurationItem]] = {}
        self.environment_configs: Dict[EnvironmentType, Dict[str, Any]] = {}
        self.config_templates: Dict[str, str] = {}

        logging.info(
            f"Configuration manager initialized with directory: {config_directory}"
        )

    def add_configuration_item(self, namespace: str, item: ConfigurationItem):
        """Add a configuration item to a namespace"""
        if namespace not in self.configurations:
            self.configurations[namespace] = {}

        self.configurations[namespace][item.key] = item
        logging.info(f"Added configuration item: {namespace}.{item.key}")

    def set_environment_config(
        self, environment: EnvironmentType, config: Dict[str, Any]
    ):
        """Set configuration for specific environment"""
        self.environment_configs[environment] = config

        # Save to file
        config_file = self.config_directory / f"{environment.value}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logging.info(f"Environment configuration set for {environment.value}")

    def get_environment_config(self, environment: EnvironmentType) -> Dict[str, Any]:
        """Get configuration for specific environment"""
        if environment in self.environment_configs:
            return self.environment_configs[environment]

        # Try to load from file
        config_file = self.config_directory / f"{environment.value}.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                self.environment_configs[environment] = config
                return config

        return {}

    def generate_config_file(
        self,
        namespace: str,
        environment: EnvironmentType,
        format: ConfigFormat = ConfigFormat.YAML,
    ) -> str:
        """Generate configuration file for specific environment"""

        if namespace not in self.configurations:
            raise ValueError(f"Configuration namespace '{namespace}' not found")

        config_items = self.configurations[namespace]
        env_config = self.get_environment_config(environment)

        # Build configuration
        config_data = {}
        for key, item in config_items.items():
            # Use environment-specific value if available
            if item.environment_specific and key in env_config:
                value = env_config[key]
            else:
                value = item.value

            # Validate value if rules exist
            if item.validation_rules:
                self._validate_config_value(key, value, item.validation_rules)

            config_data[key] = value

        # Generate file content based on format
        if format == ConfigFormat.YAML:
            content = yaml.dump(config_data, default_flow_style=False)
        elif format == ConfigFormat.JSON:
            content = json.dumps(config_data, indent=2)
        elif format == ConfigFormat.ENV:
            content = self._generate_env_format(config_data)
        else:
            raise ValueError(f"Unsupported config format: {format}")

        # Save to file
        filename = f"{namespace}_{environment.value}.{format.value}"
        config_path = self.config_directory / filename

        with open(config_path, "w") as f:
            f.write(content)

        logging.info(f"Generated configuration file: {config_path}")
        return str(config_path)

    def _validate_config_value(self, key: str, value: Any, rules: List[str]):
        """Validate configuration value against rules"""
        for rule in rules:
            if rule.startswith("type:"):
                expected_type = rule.split(":")[1]
                if expected_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Config '{key}' must be a string")
                elif expected_type == "integer" and not isinstance(value, int):
                    raise ValueError(f"Config '{key}' must be an integer")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Config '{key}' must be a boolean")

            elif rule.startswith("min:"):
                min_val = float(rule.split(":")[1])
                if isinstance(value, (int, float)) and value < min_val:
                    raise ValueError(f"Config '{key}' must be >= {min_val}")

            elif rule.startswith("max:"):
                max_val = float(rule.split(":")[1])
                if isinstance(value, (int, float)) and value > max_val:
                    raise ValueError(f"Config '{key}' must be <= {max_val}")

    def _generate_env_format(self, config_data: Dict[str, Any]) -> str:
        """Generate environment variable format"""
        lines = []
        for key, value in config_data.items():
            # Convert to uppercase and replace dots with underscores
            env_key = key.upper().replace(".", "_")
            lines.append(f"{env_key}={value}")

        return "\n".join(lines)

    def backup_configuration(self, environment: EnvironmentType) -> str:
        """Create backup of environment configuration"""
        timestamp = int(time.time())
        backup_name = f"config_backup_{environment.value}_{timestamp}"
        backup_path = self.config_directory / "backups" / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        # Copy configuration files
        for config_file in self.config_directory.glob(f"*{environment.value}*"):
            if config_file.is_file():
                shutil.copy2(config_file, backup_path)

        logging.info(f"Configuration backup created: {backup_path}")
        return str(backup_path)


class DeploymentEngine:
    """Handles deployment operations"""

    def __init__(self, deployment_directory: str = "deployments"):
        self.deployment_directory = Path(deployment_directory)
        self.deployment_directory.mkdir(exist_ok=True)

        self.deployment_records: List[DeploymentRecord] = []
        self.deployment_targets: Dict[str, DeploymentTarget] = {}
        self.deployment_strategies = {
            "local": self._deploy_local,
            "docker": self._deploy_docker,
            "kubernetes": self._deploy_kubernetes,
        }

        logging.info(
            f"Deployment engine initialized with directory: {deployment_directory}"
        )

    def register_target(self, target: DeploymentTarget):
        """Register a deployment target"""
        self.deployment_targets[target.name] = target
        logging.info(f"Deployment target registered: {target.name}")

    def create_deployment_package(
        self, package: DeploymentPackage, source_directory: str
    ) -> str:
        """Create deployment package from source directory"""

        package_path = (
            self.deployment_directory / f"{package.name}_{package.version}.zip"
        )

        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            source_path = Path(source_directory)

            # Add source files
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)

            # Add package metadata
            metadata = {
                "name": package.name,
                "version": package.version,
                "description": package.description,
                "artifacts": package.artifacts,
                "dependencies": package.dependencies,
                "metadata": package.metadata,
            }

            zipf.writestr("package.json", json.dumps(metadata, indent=2))

        logging.info(f"Deployment package created: {package_path}")
        return str(package_path)

    def deploy_package(
        self,
        package_path: str,
        target_name: str,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Deploy package to target"""

        if target_name not in self.deployment_targets:
            raise ValueError(f"Deployment target '{target_name}' not found")

        target = self.deployment_targets[target_name]
        deployment_id = f"deploy_{int(time.time())}_{target_name}"

        # Extract package information
        package_info = self._extract_package_info(package_path)

        # Create deployment record
        record = DeploymentRecord(
            deployment_id=deployment_id,
            package_name=package_info["name"],
            package_version=package_info["version"],
            target_name=target_name,
            status=DeploymentStatus.PENDING,
            start_time=time.time(),
            configuration_hash=self._calculate_config_hash(configuration or {}),
        )

        self.deployment_records.append(record)

        try:
            record.status = DeploymentStatus.IN_PROGRESS
            record.logs.append(f"Starting deployment to {target_name}")

            # Execute deployment strategy
            strategy = self.deployment_strategies.get(target.type)
            if not strategy:
                raise ValueError(f"Unsupported deployment target type: {target.type}")

            artifacts_path = strategy(package_path, target, configuration or {})
            record.artifacts_path = artifacts_path

            # Run health check if configured
            if target.health_check_url:
                self._run_health_check(target.health_check_url)

            record.status = DeploymentStatus.COMPLETED
            record.end_time = time.time()
            record.logs.append("Deployment completed successfully")

            logging.info(f"Deployment {deployment_id} completed successfully")

        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.end_time = time.time()
            record.logs.append(f"Deployment failed: {str(e)}")

            logging.error(f"Deployment {deployment_id} failed: {e}")
            raise

        return deployment_id

    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""

        record = self._get_deployment_record(deployment_id)
        if not record:
            raise ValueError(f"Deployment record '{deployment_id}' not found")

        if record.status != DeploymentStatus.COMPLETED:
            raise ValueError(f"Cannot rollback deployment with status: {record.status}")

        try:
            # Find previous successful deployment for the same target
            previous_deployment = self._find_previous_deployment(
                record.target_name, deployment_id
            )

            if previous_deployment:
                # Restore previous deployment
                target = self.deployment_targets[record.target_name]

                if previous_deployment.artifacts_path and os.path.exists(
                    previous_deployment.artifacts_path
                ):
                    self._restore_deployment(previous_deployment.artifacts_path, target)

                    record.status = DeploymentStatus.ROLLED_BACK
                    record.rollback_info = {
                        "rolled_back_to": previous_deployment.deployment_id,
                        "rollback_time": time.time(),
                    }
                    record.logs.append(
                        f"Rolled back to deployment {previous_deployment.deployment_id}"
                    )

                    logging.info(f"Deployment {deployment_id} rolled back successfully")
                    return True

            raise ValueError("No previous deployment found for rollback")

        except Exception as e:
            record.logs.append(f"Rollback failed: {str(e)}")
            logging.error(f"Rollback of deployment {deployment_id} failed: {e}")
            return False

    def _extract_package_info(self, package_path: str) -> Dict[str, Any]:
        """Extract package information"""
        with zipfile.ZipFile(package_path, "r") as zipf:
            with zipf.open("package.json") as f:
                return json.load(f)

    def _calculate_config_hash(self, configuration: Dict[str, Any]) -> str:
        """Calculate hash of configuration"""
        config_str = json.dumps(configuration, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _deploy_local(
        self, package_path: str, target: DeploymentTarget, configuration: Dict[str, Any]
    ) -> str:
        """Deploy to local filesystem"""

        deploy_path = Path(target.connection_params.get("path", "/tmp/deployments"))
        deploy_path = deploy_path / f"deployment_{int(time.time())}"
        deploy_path.mkdir(parents=True, exist_ok=True)

        # Extract package
        with zipfile.ZipFile(package_path, "r") as zipf:
            zipf.extractall(deploy_path)

        # Write configuration file
        config_file = deploy_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(configuration, f, indent=2)

        # Run startup script if exists
        startup_script = deploy_path / "startup.sh"
        if startup_script.exists():
            subprocess.run(["chmod", "+x", str(startup_script)], check=True)
            subprocess.run([str(startup_script)], cwd=deploy_path, check=True)

        return str(deploy_path)

    def _deploy_docker(
        self, package_path: str, target: DeploymentTarget, configuration: Dict[str, Any]
    ) -> str:
        """Deploy to Docker container"""

        # Create temporary directory for Docker build
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract package
            with zipfile.ZipFile(package_path, "r") as zipf:
                zipf.extractall(temp_path)

            # Create Dockerfile if not exists
            dockerfile_path = temp_path / "Dockerfile"
            if not dockerfile_path.exists():
                self._create_default_dockerfile(dockerfile_path, target)

            # Build Docker image
            image_name = (
                f"{target.connection_params.get('registry', 'local')}/agent-system"
            )
            build_cmd = ["docker", "build", "-t", image_name, str(temp_path)]
            subprocess.run(build_cmd, check=True)

            # Run Docker container
            container_name = f"agent-system-{int(time.time())}"
            run_cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "8080:8080",  # Default port mapping
                image_name,
            ]

            # Add environment variables from configuration
            for key, value in configuration.items():
                run_cmd.extend(["-e", f"{key}={value}"])

            subprocess.run(run_cmd, check=True)

            return container_name

    def _deploy_kubernetes(
        self, package_path: str, target: DeploymentTarget, configuration: Dict[str, Any]
    ) -> str:
        """Deploy to Kubernetes cluster"""

        # Create temporary directory for Kubernetes manifests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract package
            with zipfile.ZipFile(package_path, "r") as zipf:
                zipf.extractall(temp_path)

            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests(target, configuration)

            # Write manifests to files
            for name, content in manifests.items():
                manifest_file = temp_path / f"{name}.yaml"
                with open(manifest_file, "w") as f:
                    yaml.dump(content, f, default_flow_style=False)

            # Apply manifests
            namespace = target.connection_params.get("namespace", "default")
            for manifest_file in temp_path.glob("*.yaml"):
                apply_cmd = [
                    "kubectl",
                    "apply",
                    "-f",
                    str(manifest_file),
                    "-n",
                    namespace,
                ]
                subprocess.run(apply_cmd, check=True)

            return f"kubernetes-{namespace}"

    def _create_default_dockerfile(
        self, dockerfile_path: Path, target: DeploymentTarget
    ):
        """Create default Dockerfile"""

        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
"""

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content.strip())

    def _generate_k8s_manifests(
        self, target: DeploymentTarget, configuration: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Generate Kubernetes deployment manifests"""

        app_name = "agent-system"

        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": app_name, "labels": {"app": app_name}},
            "spec": {
                "replicas": target.connection_params.get("replicas", 2),
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": app_name,
                                "image": target.connection_params.get(
                                    "image", "agent-system:latest"
                                ),
                                "ports": [{"containerPort": 8080}],
                                "env": [
                                    {"name": k, "value": str(v)}
                                    for k, v in configuration.items()
                                ],
                                "resources": target.resource_limits,
                            }
                        ]
                    },
                },
            },
        }

        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{app_name}-service", "labels": {"app": app_name}},
            "spec": {
                "selector": {"app": app_name},
                "ports": [{"port": 80, "targetPort": 8080}],
                "type": "ClusterIP",
            },
        }

        return {"deployment": deployment, "service": service}

    def _run_health_check(self, health_check_url: str):
        """Run health check against deployed service"""
        import urllib.request
        import urllib.error

        max_retries = 5
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(health_check_url, timeout=10) as response:
                    if response.status == 200:
                        logging.info("Health check passed")
                        return
                    else:
                        raise Exception(
                            f"Health check failed with status: {response.status}"
                        )

            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Health check attempt {attempt + 1} failed, retrying in {retry_delay}s"
                    )
                    time.sleep(retry_delay)
                else:
                    raise Exception(
                        f"Health check failed after {max_retries} attempts: {e}"
                    )

    def _get_deployment_record(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get deployment record by ID"""
        for record in self.deployment_records:
            if record.deployment_id == deployment_id:
                return record
        return None

    def _find_previous_deployment(
        self, target_name: str, current_deployment_id: str
    ) -> Optional[DeploymentRecord]:
        """Find previous successful deployment for target"""

        # Filter deployments for the target, excluding current one
        target_deployments = [
            r
            for r in self.deployment_records
            if r.target_name == target_name
            and r.deployment_id != current_deployment_id
            and r.status == DeploymentStatus.COMPLETED
        ]

        # Sort by start time (most recent first)
        target_deployments.sort(key=lambda x: x.start_time, reverse=True)

        return target_deployments[0] if target_deployments else None

    def _restore_deployment(self, artifacts_path: str, target: DeploymentTarget):
        """Restore a previous deployment"""

        if target.type == "local":
            # For local deployments, copy artifacts back
            current_path = (
                Path(target.connection_params.get("path", "/tmp/deployments"))
                / "current"
            )
            if current_path.exists():
                shutil.rmtree(current_path)
            shutil.copytree(artifacts_path, current_path)

        elif target.type == "docker":
            # For Docker, restart with previous image/container
            # This would involve more complex Docker operations
            pass

        elif target.type == "kubernetes":
            # For Kubernetes, apply previous manifests
            # This would involve kubectl operations
            pass

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""

        record = self._get_deployment_record(deployment_id)
        if not record:
            return None

        return {
            "deployment_id": record.deployment_id,
            "package_name": record.package_name,
            "package_version": record.package_version,
            "target_name": record.target_name,
            "status": record.status.value,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "duration": (
                (record.end_time - record.start_time) if record.end_time else None
            ),
            "deployed_by": record.deployed_by,
            "logs": record.logs[-10:],  # Last 10 log entries
            "rollback_info": record.rollback_info,
        }

    def list_deployments(
        self, target_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by target"""

        deployments = self.deployment_records

        if target_name:
            deployments = [d for d in deployments if d.target_name == target_name]

        return [self.get_deployment_status(d.deployment_id) for d in deployments]


class DeploymentOrchestrator:
    """Orchestrates complex deployment workflows"""

    def __init__(
        self, config_manager: ConfigurationManager, deployment_engine: DeploymentEngine
    ):
        self.config_manager = config_manager
        self.deployment_engine = deployment_engine

        self.deployment_workflows: Dict[str, List[Dict[str, Any]]] = {}

    def create_deployment_workflow(
        self, workflow_name: str, steps: List[Dict[str, Any]]
    ):
        """Create a multi-step deployment workflow"""
        self.deployment_workflows[workflow_name] = steps
        logging.info(f"Deployment workflow created: {workflow_name}")

    def execute_workflow(
        self, workflow_name: str, package_path: str, environment: EnvironmentType
    ) -> List[str]:
        """Execute a deployment workflow"""

        if workflow_name not in self.deployment_workflows:
            raise ValueError(f"Deployment workflow '{workflow_name}' not found")

        steps = self.deployment_workflows[workflow_name]
        deployment_ids = []

        try:
            for i, step in enumerate(steps):
                step_name = step.get("name", f"Step {i+1}")
                target_name = step["target"]
                config_namespace = step.get("config_namespace", "default")

                logging.info(f"Executing workflow step: {step_name}")

                # Generate configuration for this step
                config_file = self.config_manager.generate_config_file(
                    config_namespace, environment, ConfigFormat.JSON
                )

                with open(config_file, "r") as f:
                    configuration = json.load(f)

                # Add step-specific configuration
                if "config_overrides" in step:
                    configuration.update(step["config_overrides"])

                # Deploy to target
                deployment_id = self.deployment_engine.deploy_package(
                    package_path, target_name, configuration
                )

                deployment_ids.append(deployment_id)

                # Wait for deployment completion if specified
                if step.get("wait_for_completion", True):
                    self._wait_for_deployment(deployment_id)

                # Run post-deployment scripts if specified
                if "post_deploy_script" in step:
                    self._run_script(step["post_deploy_script"], configuration)

            logging.info(
                f"Deployment workflow '{workflow_name}' completed successfully"
            )
            return deployment_ids

        except Exception as e:
            logging.error(f"Deployment workflow '{workflow_name}' failed: {e}")

            # Rollback completed deployments
            for deployment_id in reversed(deployment_ids):
                try:
                    self.deployment_engine.rollback_deployment(deployment_id)
                except Exception as rollback_error:
                    logging.error(
                        f"Rollback failed for {deployment_id}: {rollback_error}"
                    )

            raise

    def _wait_for_deployment(self, deployment_id: str, timeout: int = 300):
        """Wait for deployment to complete"""

        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.deployment_engine.get_deployment_status(deployment_id)

            if not status:
                raise ValueError(f"Deployment {deployment_id} not found")

            if status["status"] == DeploymentStatus.COMPLETED.value:
                return
            elif status["status"] == DeploymentStatus.FAILED.value:
                raise Exception(f"Deployment {deployment_id} failed")

            time.sleep(5)

        raise TimeoutError(
            f"Deployment {deployment_id} timed out after {timeout} seconds"
        )

    def _run_script(self, script_path: str, configuration: Dict[str, Any]):
        """Run post-deployment script"""

        # Set environment variables from configuration
        env = os.environ.copy()
        env.update({k: str(v) for k, v in configuration.items()})

        result = subprocess.run([script_path], env=env, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Script {script_path} failed: {result.stderr}")

        logging.info(f"Script {script_path} executed successfully")


# Example usage and factory functions
def create_sample_deployment_system() -> (
    Tuple[ConfigurationManager, DeploymentEngine, DeploymentOrchestrator]
):
    """Create a sample deployment system"""

    # Create configuration manager
    config_manager = ConfigurationManager()

    # Add sample configuration items
    config_manager.add_configuration_item(
        "agent_system",
        ConfigurationItem(
            key="log_level",
            value="INFO",
            description="Logging level",
            validation_rules=["type:string"],
        ),
    )

    config_manager.add_configuration_item(
        "agent_system",
        ConfigurationItem(
            key="max_agents",
            value=10,
            description="Maximum number of agents",
            environment_specific=True,
            validation_rules=["type:integer", "min:1", "max:100"],
        ),
    )

    config_manager.add_configuration_item(
        "agent_system",
        ConfigurationItem(
            key="database_url",
            value="postgresql://localhost:5432/agents",
            description="Database connection URL",
            sensitive=True,
            environment_specific=True,
        ),
    )

    # Set environment-specific configurations
    config_manager.set_environment_config(
        EnvironmentType.DEVELOPMENT,
        {"max_agents": 5, "database_url": "postgresql://localhost:5432/agents_dev"},
    )

    config_manager.set_environment_config(
        EnvironmentType.PRODUCTION,
        {"max_agents": 50, "database_url": "postgresql://prod-db:5432/agents_prod"},
    )

    # Create deployment engine
    deployment_engine = DeploymentEngine()

    # Register deployment targets
    deployment_engine.register_target(
        DeploymentTarget(
            name="local_dev",
            type="local",
            environment=EnvironmentType.DEVELOPMENT,
            connection_params={"path": "/tmp/agent_deployments"},
            health_check_url="http://localhost:8080/health",
        )
    )

    deployment_engine.register_target(
        DeploymentTarget(
            name="docker_staging",
            type="docker",
            environment=EnvironmentType.STAGING,
            connection_params={"registry": "staging-registry"},
            resource_limits={"memory": "512Mi", "cpu": "500m"},
        )
    )

    deployment_engine.register_target(
        DeploymentTarget(
            name="k8s_production",
            type="kubernetes",
            environment=EnvironmentType.PRODUCTION,
            connection_params={
                "namespace": "agents",
                "replicas": 3,
                "image": "production-registry/agent-system:latest",
            },
            resource_limits={"memory": "1Gi", "cpu": "1000m"},
            health_check_url="http://agents.production.com/health",
        )
    )

    # Create orchestrator
    orchestrator = DeploymentOrchestrator(config_manager, deployment_engine)

    # Create sample deployment workflow
    orchestrator.create_deployment_workflow(
        "full_deployment",
        [
            {
                "name": "Deploy to Development",
                "target": "local_dev",
                "config_namespace": "agent_system",
                "wait_for_completion": True,
            },
            {
                "name": "Deploy to Staging",
                "target": "docker_staging",
                "config_namespace": "agent_system",
                "wait_for_completion": True,
            },
            {
                "name": "Deploy to Production",
                "target": "k8s_production",
                "config_namespace": "agent_system",
                "config_overrides": {"production_mode": True},
                "wait_for_completion": True,
            },
        ],
    )

    return config_manager, deployment_engine, orchestrator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create deployment system
    config_manager, deployment_engine, orchestrator = create_sample_deployment_system()

    # Generate configuration files for different environments
    print("Generating configuration files...")

    dev_config = config_manager.generate_config_file(
        "agent_system", EnvironmentType.DEVELOPMENT
    )
    prod_config = config_manager.generate_config_file(
        "agent_system", EnvironmentType.PRODUCTION
    )

    print(f"Development config: {dev_config}")
    print(f"Production config: {prod_config}")

    # Create a sample deployment package
    print("\nCreating deployment package...")

    package = DeploymentPackage(
        name="agent-system",
        version="1.0.0",
        description="Agent system deployment package",
        artifacts=["main.py", "requirements.txt"],
        dependencies=["python>=3.9"],
    )

    # For demo purposes, create a temporary source directory
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample files
        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write("print('Agent system started')")

        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write("requests>=2.25.0\npyyaml>=5.4.0")

        package_path = deployment_engine.create_deployment_package(package, temp_dir)
        print(f"Package created: {package_path}")

        # Deploy to local development
        print("\nDeploying to local development...")
        deployment_id = deployment_engine.deploy_package(
            package_path, "local_dev", {"debug_mode": True}
        )

        print(f"Deployment ID: {deployment_id}")

        # Check deployment status
        status = deployment_engine.get_deployment_status(deployment_id)
        print(f"Deployment status: {status}")

        # List all deployments
        deployments = deployment_engine.list_deployments()
        print(f"\nAll deployments: {len(deployments)} found")

    print("\nDeployment system demonstration completed")
