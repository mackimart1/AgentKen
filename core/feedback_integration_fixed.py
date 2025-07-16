"""
Feedback Integration System (Fixed Version)
Seamlessly integrates feedback collection with existing agents and tools.
"""

import functools
import time
import uuid
import logging
import threading
from typing import Any, Callable, Dict, Optional, List

from feedback_system import (
    FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, ContinuousLearningEngine,
    TaskExecution, ComponentType, TaskOutcome, FeedbackType
)
from feedback_interface import FeedbackPromptSystem


# Global feedback system instance
_feedback_system: Optional['FeedbackSystem'] = None
_system_lock = threading.Lock()


class FeedbackSystem:
    """Main feedback system coordinator"""
    
    def __init__(self, db_path: str = "feedback_system.db"):
        self.storage = FeedbackStorage(db_path)
        self.collector = FeedbackCollector(self.storage)
        self.analyzer = FeedbackAnalyzer(self.storage)
        self.learning_engine = ContinuousLearningEngine(self.storage, self.analyzer)
        self.prompt_system = FeedbackPromptSystem(self.collector)
        
        # Register default learning strategies
        self._register_default_strategies()
        
        # Feedback collection settings
        self.auto_prompt = True
        self.prompt_strategy = 'smart'
        self.feedback_handlers: List[Callable] = []
        
        logging.info("Feedback system initialized")
    
    def _register_default_strategies(self):
        """Register default learning strategies"""
        from feedback_system import (
            parameter_tuning_strategy, 
            quality_threshold_strategy, 
            error_handling_strategy
        )
        
        self.learning_engine.register_learning_strategy("parameter_tuning", parameter_tuning_strategy)
        self.learning_engine.register_learning_strategy("quality_threshold", quality_threshold_strategy)
        self.learning_engine.register_learning_strategy("error_handling", error_handling_strategy)
    
    def add_feedback_handler(self, handler: Callable):
        """Add a feedback handler"""
        self.feedback_handlers.append(handler)
        self.collector.add_feedback_handler(handler)
    
    def set_prompt_strategy(self, strategy: str):
        """Set the feedback prompting strategy"""
        self.prompt_strategy = strategy
    
    def enable_auto_prompt(self, enabled: bool = True):
        """Enable or disable automatic feedback prompting"""
        self.auto_prompt = enabled


def initialize_feedback_system(db_path: str = "feedback_system.db") -> FeedbackSystem:
    """Initialize the global feedback system"""
    global _feedback_system
    
    with _system_lock:
        if _feedback_system is None:
            _feedback_system = FeedbackSystem(db_path)
        
        return _feedback_system


def get_feedback_system() -> Optional[FeedbackSystem]:
    """Get the global feedback system instance"""
    return _feedback_system


class FeedbackTracker:
    """Context manager for tracking task executions and collecting feedback"""
    
    def __init__(self, component_id: str, component_type: ComponentType, 
                 operation: str, user_id: str = None, session_id: str = None,
                 input_data: Dict[str, Any] = None, context: Dict[str, Any] = None):
        self.component_id = component_id
        self.component_type = component_type
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.input_data = input_data or {}
        self.context = context or {}
        
        self.execution_id = str(uuid.uuid4())
        self.start_time = None
        self.execution = None
        self.feedback_system = get_feedback_system()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.feedback_system:
            return
        
        # Determine outcome
        if exc_type is None:
            outcome = TaskOutcome.SUCCESS
            output_data = getattr(self, 'output_data', {})
        else:
            outcome = TaskOutcome.ERROR if exc_type else TaskOutcome.FAILURE
            output_data = {"error": str(exc_val) if exc_val else "Unknown error"}
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Create task execution record
        self.execution = TaskExecution(
            id=self.execution_id,
            component_id=self.component_id,
            component_type=self.component_type,
            operation=self.operation,
            input_data=self.input_data,
            output_data=output_data,
            outcome=outcome,
            execution_time=execution_time,
            timestamp=self.start_time,
            user_id=self.user_id,
            session_id=self.session_id,
            context=self.context
        )
        
        # Register execution for potential feedback
        self.feedback_system.collector.register_task_execution(self.execution)
        
        # Check if we should prompt for feedback
        if (self.feedback_system.auto_prompt and 
            self.feedback_system.prompt_system.should_prompt_for_feedback(
                self.execution, self.feedback_system.prompt_strategy)):
            
            self._prompt_for_feedback()
    
    def set_output_data(self, output_data: Dict[str, Any]):
        """Set the output data for this execution"""
        self.output_data = output_data
    
    def _prompt_for_feedback(self):
        """Prompt user for feedback"""
        if not self.user_id:
            return
        
        prompt_data = self.feedback_system.prompt_system.generate_feedback_prompt(self.execution)
        
        # Log the feedback prompt (in a real system, this would trigger UI prompts)
        logging.info(f"Feedback prompt for user {self.user_id}: {prompt_data['message']}")
        
        # Notify feedback handlers
        for handler in self.feedback_system.feedback_handlers:
            try:
                handler(prompt_data)
            except Exception as e:
                logging.error(f"Feedback handler failed: {e}")


def track_agent_execution(agent_id: str, operation: str = None, **tracker_kwargs):
    """Decorator for tracking agent executions with feedback"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user context if available
            user_id = kwargs.get('user_id')
            session_id = kwargs.get('session_id')
            
            # Try to extract user context from arguments
            if not user_id and args and isinstance(args[0], dict) and 'user_id' in args[0]:
                user_id = args[0]['user_id']
            
            if not session_id and args and isinstance(args[0], dict) and 'session_id' in args[0]:
                session_id = args[0]['session_id']
            
            # Prepare input data
            input_data = {}
            try:
                input_data = {
                    'args': [str(arg) for arg in args],
                    'kwargs': {k: str(v) for k, v in kwargs.items()}
                }
            except Exception:
                input_data = {'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            
            op_name = operation or func.__name__
            
            with FeedbackTracker(
                component_id=agent_id,
                component_type=ComponentType.AGENT,
                operation=op_name,
                user_id=user_id,
                session_id=session_id,
                input_data=input_data
            ) as tracker:
                
                result = func(*args, **kwargs)
                
                # Set output data
                try:
                    output_data = {'result': str(result)}
                except Exception:
                    output_data = {'result_type': type(result).__name__}
                
                tracker.set_output_data(output_data)
                
                return result
        
        return wrapper
    return decorator


def track_tool_execution(tool_id: str, operation: str = None, **tracker_kwargs):
    """Decorator for tracking tool executions with feedback"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user context if available
            user_id = kwargs.get('user_id')
            session_id = kwargs.get('session_id')
            
            # Prepare input data
            input_data = {}
            try:
                input_data = {
                    'args': [str(arg) for arg in args],
                    'kwargs': {k: str(v) for k, v in kwargs.items()}
                }
            except Exception:
                input_data = {'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            
            op_name = operation or func.__name__
            
            with FeedbackTracker(
                component_id=tool_id,
                component_type=ComponentType.TOOL,
                operation=op_name,
                user_id=user_id,
                session_id=session_id,
                input_data=input_data
            ) as tracker:
                
                result = func(*args, **kwargs)
                
                # Set output data
                try:
                    output_data = {'result': str(result)}
                except Exception:
                    output_data = {'result_type': type(result).__name__}
                
                tracker.set_output_data(output_data)
                
                return result
        
        return wrapper
    return decorator


# Convenience functions for manual feedback collection
def collect_rating_feedback(task_execution_id: str, user_id: str, rating: float,
                          text_feedback: str = None, tags: List[str] = None):
    """Collect rating feedback"""
    system = get_feedback_system()
    if not system:
        raise RuntimeError("Feedback system not initialized")
    
    return system.collector.collect_feedback(
        task_execution_id=task_execution_id,
        user_id=user_id,
        feedback_type=FeedbackType.RATING,
        rating=rating,
        text_feedback=text_feedback,
        tags=tags or []
    )


def collect_binary_feedback(task_execution_id: str, user_id: str, satisfied: bool,
                          text_feedback: str = None, tags: List[str] = None):
    """Collect binary (thumbs up/down) feedback"""
    system = get_feedback_system()
    if not system:
        raise RuntimeError("Feedback system not initialized")
    
    return system.collector.collect_feedback(
        task_execution_id=task_execution_id,
        user_id=user_id,
        feedback_type=FeedbackType.BINARY,
        binary_score=satisfied,
        text_feedback=text_feedback,
        tags=tags or []
    )


def collect_text_feedback(task_execution_id: str, user_id: str, text_feedback: str,
                        tags: List[str] = None):
    """Collect text feedback"""
    system = get_feedback_system()
    if not system:
        raise RuntimeError("Feedback system not initialized")
    
    return system.collector.collect_feedback(
        task_execution_id=task_execution_id,
        user_id=user_id,
        feedback_type=FeedbackType.TEXT,
        text_feedback=text_feedback,
        tags=tags or []
    )


def collect_detailed_feedback(task_execution_id: str, user_id: str,
                            detailed_scores: Dict[str, float],
                            text_feedback: str = None, tags: List[str] = None):
    """Collect detailed aspect-based feedback"""
    system = get_feedback_system()
    if not system:
        raise RuntimeError("Feedback system not initialized")
    
    return system.collector.collect_feedback(
        task_execution_id=task_execution_id,
        user_id=user_id,
        feedback_type=FeedbackType.DETAILED,
        detailed_scores=detailed_scores,
        text_feedback=text_feedback,
        tags=tags or []
    )


# Feedback prompt handlers
def console_feedback_prompt_handler(prompt_data: Dict[str, Any]):
    """Console-based feedback prompt handler"""
    print(f"\n📝 Feedback Request:")
    print(f"Task: {prompt_data['component_name']} - {prompt_data['operation']}")
    print(f"Message: {prompt_data['message']}")
    print(f"Task ID: {prompt_data['task_id']}")
    print(f"Suggested feedback type: {prompt_data['suggested_feedback_type']}")
    print("---")


def web_feedback_prompt_handler(prompt_data: Dict[str, Any]):
    """Web-based feedback prompt handler"""
    # In a real implementation, this would trigger a web notification or popup
    logging.info(f"Web feedback prompt: {prompt_data['message']} (Task: {prompt_data['task_id']})")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize feedback system
    feedback_system = initialize_feedback_system("test_feedback_integration.db")
    
    # Add feedback prompt handlers
    feedback_system.add_feedback_handler(console_feedback_prompt_handler)
    
    # Example: Using decorators
    @track_agent_execution("example_agent", "search")
    def search_web(query: str, user_id: str = None) -> Dict[str, Any]:
        time.sleep(0.1)  # Simulate work
        return {"results": [f"Result for {query}"]}
    
    @track_tool_execution("data_processor", "transform")
    def transform_data(data: List[str], user_id: str = None) -> List[str]:
        time.sleep(0.05)  # Simulate work
        return [item.upper() for item in data]
    
    # Execute some operations
    print("Executing operations with feedback tracking...")
    
    # Test decorator approach
    search_result = search_web("AI frameworks", user_id="user123")
    transform_result = transform_data(["hello", "world"], user_id="user123")
    
    # Simulate manual feedback collection
    print("\nCollecting manual feedback...")
    
    # Get recent tasks for user
    recent_tasks = feedback_system.collector.get_pending_tasks_for_user("user123")
    if recent_tasks:
        task = recent_tasks[0]
        
        # Collect rating feedback
        rating_feedback = collect_rating_feedback(
            task_execution_id=task.id,
            user_id="user123",
            rating=4.5,
            text_feedback="Great results!",
            tags=["helpful", "fast"]
        )
        
        print(f"Collected rating feedback: {rating_feedback.id}")
    
    # Analyze performance
    print("\nAnalyzing performance...")
    profile = feedback_system.analyzer.analyze_component_performance("example_agent")
    print(f"Average rating: {profile.average_rating:.2f}")
    print(f"Satisfaction rate: {profile.satisfaction_rate:.1f}%")
    
    # Generate insights and apply learning
    print("\nGenerating insights and applying learning...")
    insights = feedback_system.analyzer.generate_learning_insights()
    adaptations = feedback_system.learning_engine.apply_learning()
    
    print(f"Generated {len(insights)} insights")
    print(f"Applied {len(adaptations)} learning adaptations")
    
    print("\nFeedback integration demonstration complete!")