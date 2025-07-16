"""
Module Composition Framework
Enables flexible composition of agents and tools into workflows and complex systems.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import asyncio
import threading
from collections import defaultdict

from module_system import (
    ModuleRegistry, ModuleInterface, ModuleMetadata, ModuleType, ModuleStatus,
    get_module_registry
)


class CompositionType(Enum):
    """Types of module compositions"""
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    WORKFLOW = "workflow"
    ENSEMBLE = "ensemble"


class ExecutionMode(Enum):
    """Execution modes for compositions"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class CompositionStep:
    """Represents a step in a composition"""
    id: str
    module_id: str
    capability: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    parallel_group: Optional[str] = None


@dataclass
class CompositionMetadata:
    """Metadata for a composition"""
    id: str
    name: str
    version: str
    description: str
    composition_type: CompositionType
    execution_mode: ExecutionMode
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class ExecutionContext:
    """Context for composition execution"""
    composition_id: str
    execution_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    error: Optional[str] = None


class CompositionInterface(ABC):
    """Base interface for all compositions"""
    
    def __init__(self, composition_id: str, metadata: CompositionMetadata):
        self.composition_id = composition_id
        self.metadata = metadata
        self.registry = get_module_registry()
        self.logger = logging.getLogger(f"composition.{composition_id}")
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        """Execute the composition with given input data."""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate the composition and return any errors."""
        pass
    
    def get_metadata(self) -> CompositionMetadata:
        """Get composition metadata."""
        return self.metadata
    
    def get_required_modules(self) -> List[str]:
        """Get list of required module IDs."""
        return []


class PipelineComposition(CompositionInterface):
    """Sequential pipeline composition"""
    
    def __init__(self, composition_id: str, metadata: CompositionMetadata, steps: List[CompositionStep]):
        super().__init__(composition_id, metadata)
        self.steps = steps
    
    def execute(self, input_data: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        """Execute pipeline steps sequentially."""
        if not context:
            context = ExecutionContext(
                composition_id=self.composition_id,
                execution_id=str(uuid.uuid4()),
                input_data=input_data.copy()
            )
        
        try:
            current_data = input_data.copy()
            
            for i, step in enumerate(self.steps):
                self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.id}")
                
                # Prepare step input
                step_input = self._prepare_step_input(step, current_data, context)
                
                # Execute step
                step_output = self._execute_step(step, step_input, context)
                
                # Process step output
                current_data = self._process_step_output(step, step_output, current_data, context)
                
                # Store intermediate data
                context.intermediate_data[step.id] = step_output
            
            context.output_data = current_data
            context.status = "completed"
            context.end_time = time.time()
            
            return current_data
            
        except Exception as e:
            context.status = "failed"
            context.error = str(e)
            context.end_time = time.time()
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate pipeline composition."""
        errors = []
        
        if not self.steps:
            errors.append("Pipeline has no steps")
            return errors
        
        for i, step in enumerate(self.steps):
            # Check if module exists
            if not self.registry.get_module(step.module_id):
                errors.append(f"Step {i+1}: Module '{step.module_id}' not found")
                continue
            
            # Check if capability exists
            metadata = self.registry.get_metadata(step.module_id)
            if metadata:
                capabilities = [cap.name for cap in metadata.capabilities]
                if step.capability not in capabilities:
                    errors.append(f"Step {i+1}: Capability '{step.capability}' not found in module '{step.module_id}'")
        
        return errors
    
    def get_required_modules(self) -> List[str]:
        """Get required module IDs."""
        return [step.module_id for step in self.steps]
    
    def _prepare_step_input(self, step: CompositionStep, current_data: Dict[str, Any], 
                           context: ExecutionContext) -> Dict[str, Any]:
        """Prepare input data for a step."""
        step_input = step.parameters.copy()
        
        # Apply input mapping
        for target_key, source_key in step.input_mapping.items():
            if source_key in current_data:
                step_input[target_key] = current_data[source_key]
            elif source_key in context.intermediate_data:
                # Look in intermediate data from previous steps
                for step_id, step_data in context.intermediate_data.items():
                    if isinstance(step_data, dict) and source_key in step_data:
                        step_input[target_key] = step_data[source_key]
                        break
        
        # If no mapping specified, pass all current data
        if not step.input_mapping:
            step_input.update(current_data)
        
        return step_input
    
    def _execute_step(self, step: CompositionStep, step_input: Dict[str, Any], 
                     context: ExecutionContext) -> Any:
        """Execute a single step."""
        try:
            # Execute with timeout if specified
            if step.timeout:
                # TODO: Implement timeout handling
                pass
            
            result = self.registry.execute_capability(step.module_id, step.capability, **step_input)
            return result
            
        except Exception as e:
            # Retry if configured
            if step.retry_count > 0:
                for attempt in range(step.retry_count):
                    try:
                        self.logger.info(f"Retrying step {step.id}, attempt {attempt + 1}")
                        result = self.registry.execute_capability(step.module_id, step.capability, **step_input)
                        return result
                    except Exception:
                        if attempt == step.retry_count - 1:
                            raise
                        time.sleep(1)  # Wait before retry
            raise
    
    def _process_step_output(self, step: CompositionStep, step_output: Any, 
                           current_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Process step output and update current data."""
        if step.output_mapping:
            # Apply output mapping
            for source_key, target_key in step.output_mapping.items():
                if isinstance(step_output, dict) and source_key in step_output:
                    current_data[target_key] = step_output[source_key]
                elif source_key == "_result":
                    current_data[target_key] = step_output
        else:
            # If no mapping, merge output if it's a dict
            if isinstance(step_output, dict):
                current_data.update(step_output)
            else:
                current_data["result"] = step_output
        
        return current_data


class ParallelComposition(CompositionInterface):
    """Parallel execution composition"""
    
    def __init__(self, composition_id: str, metadata: CompositionMetadata, steps: List[CompositionStep]):
        super().__init__(composition_id, metadata)
        self.steps = steps
    
    def execute(self, input_data: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        """Execute steps in parallel."""
        if not context:
            context = ExecutionContext(
                composition_id=self.composition_id,
                execution_id=str(uuid.uuid4()),
                input_data=input_data.copy()
            )
        
        try:
            # Group steps by parallel group
            groups = defaultdict(list)
            for step in self.steps:
                group_key = step.parallel_group or "default"
                groups[group_key].append(step)
            
            results = {}
            
            # Execute each group in parallel
            for group_key, group_steps in groups.items():
                if len(group_steps) == 1:
                    # Single step, execute directly
                    step = group_steps[0]
                    step_input = self._prepare_step_input(step, input_data, context)
                    results[step.id] = self._execute_step(step, step_input, context)
                else:
                    # Multiple steps, execute in parallel
                    group_results = self._execute_parallel_group(group_steps, input_data, context)
                    results.update(group_results)
            
            # Combine results
            output_data = input_data.copy()
            for step_id, result in results.items():
                step = next(s for s in self.steps if s.id == step_id)
                output_data = self._process_step_output(step, result, output_data, context)
                context.intermediate_data[step_id] = result
            
            context.output_data = output_data
            context.status = "completed"
            context.end_time = time.time()
            
            return output_data
            
        except Exception as e:
            context.status = "failed"
            context.error = str(e)
            context.end_time = time.time()
            self.logger.error(f"Parallel execution failed: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate parallel composition."""
        errors = []
        
        if not self.steps:
            errors.append("Parallel composition has no steps")
            return errors
        
        for i, step in enumerate(self.steps):
            # Check if module exists
            if not self.registry.get_module(step.module_id):
                errors.append(f"Step {i+1}: Module '{step.module_id}' not found")
        
        return errors
    
    def get_required_modules(self) -> List[str]:
        """Get required module IDs."""
        return [step.module_id for step in self.steps]
    
    def _execute_parallel_group(self, steps: List[CompositionStep], input_data: Dict[str, Any], 
                               context: ExecutionContext) -> Dict[str, Any]:
        """Execute a group of steps in parallel."""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(steps)) as executor:
            # Submit all steps
            future_to_step = {}
            for step in steps:
                step_input = self._prepare_step_input(step, input_data, context)
                future = executor.submit(self._execute_step, step, step_input, context)
                future_to_step[future] = step
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()
                    results[step.id] = result
                except Exception as e:
                    self.logger.error(f"Step {step.id} failed: {e}")
                    raise
        
        return results
    
    def _prepare_step_input(self, step: CompositionStep, input_data: Dict[str, Any], 
                           context: ExecutionContext) -> Dict[str, Any]:
        """Prepare input data for a step."""
        step_input = step.parameters.copy()
        
        # Apply input mapping
        for target_key, source_key in step.input_mapping.items():
            if source_key in input_data:
                step_input[target_key] = input_data[source_key]
        
        # If no mapping specified, pass all input data
        if not step.input_mapping:
            step_input.update(input_data)
        
        return step_input
    
    def _execute_step(self, step: CompositionStep, step_input: Dict[str, Any], 
                     context: ExecutionContext) -> Any:
        """Execute a single step."""
        return self.registry.execute_capability(step.module_id, step.capability, **step_input)
    
    def _process_step_output(self, step: CompositionStep, step_output: Any, 
                           current_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Process step output and update current data."""
        if step.output_mapping:
            # Apply output mapping
            for source_key, target_key in step.output_mapping.items():
                if isinstance(step_output, dict) and source_key in step_output:
                    current_data[target_key] = step_output[source_key]
                elif source_key == "_result":
                    current_data[target_key] = step_output
        else:
            # If no mapping, use step ID as key
            current_data[step.id] = step_output
        
        return current_data


class ConditionalComposition(CompositionInterface):
    """Conditional execution composition"""
    
    def __init__(self, composition_id: str, metadata: CompositionMetadata, 
                 condition_step: CompositionStep, true_steps: List[CompositionStep], 
                 false_steps: List[CompositionStep] = None):
        super().__init__(composition_id, metadata)
        self.condition_step = condition_step
        self.true_steps = true_steps
        self.false_steps = false_steps or []
    
    def execute(self, input_data: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        """Execute conditional composition."""
        if not context:
            context = ExecutionContext(
                composition_id=self.composition_id,
                execution_id=str(uuid.uuid4()),
                input_data=input_data.copy()
            )
        
        try:
            # Execute condition step
            condition_input = self._prepare_step_input(self.condition_step, input_data, context)
            condition_result = self._execute_step(self.condition_step, condition_input, context)
            
            # Determine which branch to execute
            condition_value = self._evaluate_condition(condition_result)
            
            # Execute appropriate branch
            if condition_value:
                steps_to_execute = self.true_steps
                self.logger.info("Executing TRUE branch")
            else:
                steps_to_execute = self.false_steps
                self.logger.info("Executing FALSE branch")
            
            # Execute selected steps as pipeline
            if steps_to_execute:
                pipeline = PipelineComposition(
                    f"{self.composition_id}_branch",
                    CompositionMetadata(
                        id=f"{self.composition_id}_branch",
                        name="Conditional Branch",
                        version="1.0.0",
                        description="Conditional branch execution",
                        composition_type=CompositionType.PIPELINE,
                        execution_mode=self.metadata.execution_mode
                    ),
                    steps_to_execute
                )
                result = pipeline.execute(input_data, context)
            else:
                result = input_data.copy()
            
            context.output_data = result
            context.status = "completed"
            context.end_time = time.time()
            
            return result
            
        except Exception as e:
            context.status = "failed"
            context.error = str(e)
            context.end_time = time.time()
            self.logger.error(f"Conditional execution failed: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate conditional composition."""
        errors = []
        
        # Validate condition step
        if not self.registry.get_module(self.condition_step.module_id):
            errors.append(f"Condition step: Module '{self.condition_step.module_id}' not found")
        
        # Validate true branch
        if not self.true_steps:
            errors.append("Conditional composition has no true branch steps")
        
        all_steps = [self.condition_step] + self.true_steps + self.false_steps
        for i, step in enumerate(all_steps):
            if not self.registry.get_module(step.module_id):
                errors.append(f"Step {i+1}: Module '{step.module_id}' not found")
        
        return errors
    
    def get_required_modules(self) -> List[str]:
        """Get required module IDs."""
        all_steps = [self.condition_step] + self.true_steps + self.false_steps
        return [step.module_id for step in all_steps]
    
    def _evaluate_condition(self, condition_result: Any) -> bool:
        """Evaluate condition result to boolean."""
        if isinstance(condition_result, bool):
            return condition_result
        elif isinstance(condition_result, dict):
            # Look for common boolean keys
            for key in ["result", "success", "condition", "value"]:
                if key in condition_result:
                    return bool(condition_result[key])
            # If no boolean key found, check if dict is non-empty
            return bool(condition_result)
        elif isinstance(condition_result, (int, float)):
            return condition_result > 0
        elif isinstance(condition_result, str):
            return condition_result.lower() in ["true", "yes", "1", "success"]
        else:
            return bool(condition_result)
    
    def _prepare_step_input(self, step: CompositionStep, input_data: Dict[str, Any], 
                           context: ExecutionContext) -> Dict[str, Any]:
        """Prepare input data for a step."""
        step_input = step.parameters.copy()
        
        # Apply input mapping
        for target_key, source_key in step.input_mapping.items():
            if source_key in input_data:
                step_input[target_key] = input_data[source_key]
        
        # If no mapping specified, pass all input data
        if not step.input_mapping:
            step_input.update(input_data)
        
        return step_input
    
    def _execute_step(self, step: CompositionStep, step_input: Dict[str, Any], 
                     context: ExecutionContext) -> Any:
        """Execute a single step."""
        return self.registry.execute_capability(step.module_id, step.capability, **step_input)


class CompositionRegistry:
    """Registry for managing compositions"""
    
    def __init__(self):
        self.compositions: Dict[str, CompositionInterface] = {}
        self.metadata: Dict[str, CompositionMetadata] = {}
        self._lock = threading.RLock()
    
    def register_composition(self, composition: CompositionInterface) -> bool:
        """Register a composition."""
        with self._lock:
            try:
                composition_id = composition.composition_id
                metadata = composition.get_metadata()
                
                # Validate composition
                errors = composition.validate()
                if errors:
                    logging.error(f"Composition validation failed: {errors}")
                    return False
                
                # Register
                self.compositions[composition_id] = composition
                self.metadata[composition_id] = metadata
                
                logging.info(f"Composition {composition_id} registered successfully")
                return True
                
            except Exception as e:
                logging.error(f"Failed to register composition: {e}")
                return False
    
    def unregister_composition(self, composition_id: str) -> bool:
        """Unregister a composition."""
        with self._lock:
            if composition_id in self.compositions:
                del self.compositions[composition_id]
                del self.metadata[composition_id]
                logging.info(f"Composition {composition_id} unregistered")
                return True
            return False
    
    def get_composition(self, composition_id: str) -> Optional[CompositionInterface]:
        """Get a composition by ID."""
        with self._lock:
            return self.compositions.get(composition_id)
    
    def list_compositions(self, composition_type: Optional[CompositionType] = None) -> List[CompositionMetadata]:
        """List compositions with optional filtering."""
        with self._lock:
            compositions = list(self.metadata.values())
            
            if composition_type:
                compositions = [c for c in compositions if c.composition_type == composition_type]
            
            return sorted(compositions, key=lambda c: c.name)
    
    def execute_composition(self, composition_id: str, input_data: Dict[str, Any]) -> Any:
        """Execute a composition."""
        with self._lock:
            composition = self.compositions.get(composition_id)
            if not composition:
                raise ValueError(f"Composition {composition_id} not found")
            
            return composition.execute(input_data)


class CompositionBuilder:
    """Builder for creating compositions"""
    
    def __init__(self, composition_id: str, name: str, composition_type: CompositionType):
        self.composition_id = composition_id
        self.metadata = CompositionMetadata(
            id=composition_id,
            name=name,
            version="1.0.0",
            description="",
            composition_type=composition_type,
            execution_mode=ExecutionMode.SYNCHRONOUS
        )
        self.steps: List[CompositionStep] = []
        self.condition_step: Optional[CompositionStep] = None
        self.true_steps: List[CompositionStep] = []
        self.false_steps: List[CompositionStep] = []
    
    def set_description(self, description: str) -> 'CompositionBuilder':
        """Set composition description."""
        self.metadata.description = description
        return self
    
    def set_version(self, version: str) -> 'CompositionBuilder':
        """Set composition version."""
        self.metadata.version = version
        return self
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'CompositionBuilder':
        """Set execution mode."""
        self.metadata.execution_mode = mode
        return self
    
    def add_step(self, step_id: str, module_id: str, capability: str, 
                parameters: Dict[str, Any] = None, input_mapping: Dict[str, str] = None,
                output_mapping: Dict[str, str] = None, **kwargs) -> 'CompositionBuilder':
        """Add a step to the composition."""
        step = CompositionStep(
            id=step_id,
            module_id=module_id,
            capability=capability,
            parameters=parameters or {},
            input_mapping=input_mapping or {},
            output_mapping=output_mapping or {},
            **kwargs
        )
        self.steps.append(step)
        return self
    
    def set_condition(self, step_id: str, module_id: str, capability: str,
                     parameters: Dict[str, Any] = None) -> 'CompositionBuilder':
        """Set condition step for conditional composition."""
        self.condition_step = CompositionStep(
            id=step_id,
            module_id=module_id,
            capability=capability,
            parameters=parameters or {}
        )
        return self
    
    def add_true_step(self, step_id: str, module_id: str, capability: str,
                     parameters: Dict[str, Any] = None) -> 'CompositionBuilder':
        """Add step to true branch of conditional composition."""
        step = CompositionStep(
            id=step_id,
            module_id=module_id,
            capability=capability,
            parameters=parameters or {}
        )
        self.true_steps.append(step)
        return self
    
    def add_false_step(self, step_id: str, module_id: str, capability: str,
                      parameters: Dict[str, Any] = None) -> 'CompositionBuilder':
        """Add step to false branch of conditional composition."""
        step = CompositionStep(
            id=step_id,
            module_id=module_id,
            capability=capability,
            parameters=parameters or {}
        )
        self.false_steps.append(step)
        return self
    
    def build(self) -> CompositionInterface:
        """Build the composition."""
        if self.metadata.composition_type == CompositionType.PIPELINE:
            return PipelineComposition(self.composition_id, self.metadata, self.steps)
        elif self.metadata.composition_type == CompositionType.PARALLEL:
            return ParallelComposition(self.composition_id, self.metadata, self.steps)
        elif self.metadata.composition_type == CompositionType.CONDITIONAL:
            if not self.condition_step:
                raise ValueError("Conditional composition requires a condition step")
            return ConditionalComposition(
                self.composition_id, self.metadata, 
                self.condition_step, self.true_steps, self.false_steps
            )
        else:
            raise ValueError(f"Unsupported composition type: {self.metadata.composition_type}")


# Global composition registry
_composition_registry: Optional[CompositionRegistry] = None
_composition_lock = threading.Lock()


def get_composition_registry() -> CompositionRegistry:
    """Get the global composition registry."""
    global _composition_registry
    
    with _composition_lock:
        if _composition_registry is None:
            _composition_registry = CompositionRegistry()
        return _composition_registry


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple pipeline composition
    builder = CompositionBuilder("example_pipeline", "Example Pipeline", CompositionType.PIPELINE)
    
    composition = (builder
                  .set_description("Example pipeline composition")
                  .add_step("step1", "module1", "capability1", {"param1": "value1"})
                  .add_step("step2", "module2", "capability2", {"param2": "value2"})
                  .build())
    
    # Register composition
    registry = get_composition_registry()
    registry.register_composition(composition)
    
    print("Composition system initialized successfully")