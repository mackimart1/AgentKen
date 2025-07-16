from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import datetime


class AnalyzePlanInput(BaseModel):
    plan_text: str = Field(description="The plan text to analyze")
    execution_feedback: List[Dict[str, Any]] = Field(description="Execution feedback from previous steps")
    goal: str = Field(description="The original goal")


class AdaptPlanInput(BaseModel):
    original_plan: str = Field(description="Original plan text")
    failure_reason: str = Field(description="Reason for plan failure")
    failed_step: str = Field(description="The step that failed")
    adaptation_strategy: str = Field(description="Adaptation strategy to apply")


class CreatePlanInput(BaseModel):
    goal: str = Field(description="The goal to create a plan for")
    constraints: List[str] = Field(default=[], description="Any constraints to consider")
    available_agents: List[str] = Field(default=[], description="Available agents")
    priority: str = Field(default="NORMAL", description="Plan priority")


# Global plan storage
_plan_storage: Dict[str, Dict[str, Any]] = {}
_adaptation_history: List[Dict[str, Any]] = []


def _generate_plan_id() -> str:
    """Generate a unique plan ID."""
    return f"plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(_plan_storage)}"


@tool(args_schema=CreatePlanInput)
def create_adaptive_plan(
    goal: str,
    constraints: List[str] = None,
    available_agents: List[str] = None,
    priority: str = "NORMAL"
) -> str:
    """
    Create an adaptive execution plan for a given goal.
    
    Args:
        goal: The goal to create a plan for
        constraints: Any constraints to consider
        available_agents: Available agents
        priority: Plan priority
    
    Returns:
        JSON string with plan creation result
    """
    try:
        if constraints is None:
            constraints = []
        if available_agents is None:
            available_agents = []
        
        plan_id = _generate_plan_id()
        
        # Create basic plan structure
        plan = {
            "id": plan_id,
            "goal": goal,
            "priority": priority,
            "status": "active",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "constraints": constraints,
            "available_agents": available_agents,
            "steps": [],
            "adaptation_count": 0,
            "success_probability": 0.8,  # Initial estimate
            "estimated_duration": None,
            "actual_duration": None,
            "execution_feedback": []
        }
        
        # Generate initial steps based on goal analysis
        steps = _generate_plan_steps(goal, available_agents, constraints)
        plan["steps"] = steps
        plan["estimated_duration"] = sum(step.get("estimated_duration", 10) for step in steps)
        
        _plan_storage[plan_id] = plan
        
        return json.dumps({
            "status": "success",
            "plan_id": plan_id,
            "plan": plan,
            "message": f"Adaptive plan created with {len(steps)} steps"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to create adaptive plan: {str(e)}"
        })


@tool(args_schema=AnalyzePlanInput)
def analyze_plan_execution(
    plan_text: str,
    execution_feedback: List[Dict[str, Any]],
    goal: str
) -> str:
    """
    Analyze plan execution and provide adaptation recommendations.
    
    Args:
        plan_text: The plan text to analyze
        execution_feedback: Execution feedback from previous steps
        goal: The original goal
    
    Returns:
        JSON string with analysis results
    """
    try:
        analysis = {
            "success_rate": 0.0,
            "failure_points": [],
            "bottlenecks": [],
            "adaptation_recommendations": [],
            "risk_assessment": "low",
            "confidence": 0.0
        }
        
        # Analyze execution feedback
        total_steps = len(execution_feedback)
        successful_steps = sum(1 for feedback in execution_feedback if feedback.get("status") == "success")
        
        if total_steps > 0:
            analysis["success_rate"] = successful_steps / total_steps
        
        # Identify failure points
        for i, feedback in enumerate(execution_feedback):
            if feedback.get("status") == "failure":
                analysis["failure_points"].append({
                    "step_index": i,
                    "error": feedback.get("message", "Unknown error"),
                    "step_description": feedback.get("step_description", "Unknown step")
                })
        
        # Identify bottlenecks (steps taking too long)
        for i, feedback in enumerate(execution_feedback):
            duration = feedback.get("duration", 0)
            if duration > 300:  # More than 5 minutes
                analysis["bottlenecks"].append({
                    "step_index": i,
                    "duration": duration,
                    "step_description": feedback.get("step_description", "Unknown step")
                })
        
        # Generate adaptation recommendations
        if analysis["success_rate"] < 0.5:
            analysis["adaptation_recommendations"].append({
                "type": "major_revision",
                "description": "Plan has low success rate, consider major revision",
                "priority": "high"
            })
            analysis["risk_assessment"] = "high"
        elif analysis["failure_points"]:
            for failure in analysis["failure_points"]:
                if "agent not found" in failure["error"].lower():
                    analysis["adaptation_recommendations"].append({
                        "type": "agent_substitution",
                        "description": f"Replace unavailable agent in step {failure['step_index']}",
                        "priority": "medium"
                    })
                elif "timeout" in failure["error"].lower():
                    analysis["adaptation_recommendations"].append({
                        "type": "step_decomposition",
                        "description": f"Break down complex step {failure['step_index']}",
                        "priority": "medium"
                    })
        
        if analysis["bottlenecks"]:
            analysis["adaptation_recommendations"].append({
                "type": "optimization",
                "description": "Optimize slow-performing steps",
                "priority": "low"
            })
        
        # Calculate confidence based on data quality
        analysis["confidence"] = min(1.0, total_steps / 5.0)  # Higher confidence with more data
        
        return json.dumps({
            "status": "success",
            "analysis": analysis,
            "message": f"Plan analysis completed with {len(analysis['adaptation_recommendations'])} recommendations"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to analyze plan execution: {str(e)}"
        })


@tool(args_schema=AdaptPlanInput)
def adapt_plan(
    original_plan: str,
    failure_reason: str,
    failed_step: str,
    adaptation_strategy: str
) -> str:
    """
    Adapt a plan based on execution failures.
    
    Args:
        original_plan: Original plan text
        failure_reason: Reason for plan failure
        failed_step: The step that failed
        adaptation_strategy: Adaptation strategy to apply
    
    Returns:
        JSON string with adapted plan
    """
    try:
        adaptation_id = f"adapt_{datetime.datetime.now().timestamp()}"
        
        # Parse adaptation strategy
        adaptations = {
            "agent_substitution": _apply_agent_substitution,
            "step_decomposition": _apply_step_decomposition,
            "dependency_reorder": _apply_dependency_reorder,
            "step_modification": _apply_step_modification,
            "parallel_execution": _apply_parallel_execution,
            "fallback_strategy": _apply_fallback_strategy
        }
        
        adaptation_func = adaptations.get(adaptation_strategy, _apply_step_modification)
        
        # Apply adaptation
        adapted_plan = adaptation_func(original_plan, failed_step, failure_reason)
        
        # Record adaptation
        adaptation_record = {
            "id": adaptation_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_plan": original_plan,
            "adapted_plan": adapted_plan,
            "failure_reason": failure_reason,
            "failed_step": failed_step,
            "strategy": adaptation_strategy,
            "success": None  # To be updated later
        }
        
        _adaptation_history.append(adaptation_record)
        
        return json.dumps({
            "status": "success",
            "adaptation_id": adaptation_id,
            "adapted_plan": adapted_plan,
            "strategy_applied": adaptation_strategy,
            "message": f"Plan adapted using {adaptation_strategy} strategy"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to adapt plan: {str(e)}"
        })


@tool
def get_adaptation_history() -> str:
    """
    Get the history of plan adaptations.
    
    Returns:
        JSON string with adaptation history
    """
    try:
        # Calculate success rates for different strategies
        strategy_stats = {}
        for adaptation in _adaptation_history:
            strategy = adaptation["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0}
            
            strategy_stats[strategy]["total"] += 1
            if adaptation.get("success"):
                strategy_stats[strategy]["successful"] += 1
        
        # Calculate success rates
        for strategy, stats in strategy_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
            else:
                stats["success_rate"] = 0.0
        
        return json.dumps({
            "status": "success",
            "adaptation_history": _adaptation_history,
            "strategy_statistics": strategy_stats,
            "total_adaptations": len(_adaptation_history),
            "message": f"Retrieved {len(_adaptation_history)} adaptation records"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get adaptation history: {str(e)}"
        })


@tool
def update_adaptation_success(adaptation_id: str, success: bool) -> str:
    """
    Update the success status of a plan adaptation.
    
    Args:
        adaptation_id: ID of the adaptation to update
        success: Whether the adaptation was successful
    
    Returns:
        JSON string with update result
    """
    try:
        for adaptation in _adaptation_history:
            if adaptation["id"] == adaptation_id:
                adaptation["success"] = success
                adaptation["updated_at"] = datetime.datetime.now().isoformat()
                
                return json.dumps({
                    "status": "success",
                    "message": f"Adaptation {adaptation_id} marked as {'successful' if success else 'failed'}"
                })
        
        return json.dumps({
            "status": "failure",
            "message": f"Adaptation {adaptation_id} not found"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to update adaptation success: {str(e)}"
        })


# Helper functions for different adaptation strategies

def _generate_plan_steps(goal: str, available_agents: List[str], constraints: List[str]) -> List[Dict[str, Any]]:
    """Generate initial plan steps based on goal analysis."""
    # This is a simplified implementation
    # In a real system, this would use more sophisticated planning algorithms
    
    steps = []
    
    # Basic step generation based on common patterns
    if "create" in goal.lower() or "build" in goal.lower():
        steps.extend([
            {
                "id": "step_1",
                "description": "Analyze requirements and gather resources",
                "agent": "web_researcher" if "web_researcher" in available_agents else None,
                "estimated_duration": 15,
                "dependencies": []
            },
            {
                "id": "step_2", 
                "description": "Design and plan implementation",
                "agent": "software_engineer" if "software_engineer" in available_agents else None,
                "estimated_duration": 30,
                "dependencies": ["step_1"]
            },
            {
                "id": "step_3",
                "description": "Implement and test solution",
                "agent": "software_engineer" if "software_engineer" in available_agents else None,
                "estimated_duration": 45,
                "dependencies": ["step_2"]
            }
        ])
    elif "research" in goal.lower() or "find" in goal.lower():
        steps.extend([
            {
                "id": "step_1",
                "description": "Conduct initial research",
                "agent": "web_researcher" if "web_researcher" in available_agents else None,
                "estimated_duration": 20,
                "dependencies": []
            },
            {
                "id": "step_2",
                "description": "Analyze and synthesize findings",
                "agent": "web_researcher" if "web_researcher" in available_agents else None,
                "estimated_duration": 15,
                "dependencies": ["step_1"]
            }
        ])
    else:
        # Generic steps
        steps.append({
            "id": "step_1",
            "description": f"Execute task: {goal}",
            "agent": available_agents[0] if available_agents else None,
            "estimated_duration": 30,
            "dependencies": []
        })
    
    return steps


def _apply_agent_substitution(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply agent substitution adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Try alternative agent due to: {failure_reason}]"
    )
    return adapted_plan


def _apply_step_decomposition(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply step decomposition adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Break into smaller sub-steps due to: {failure_reason}]"
    )
    return adapted_plan


def _apply_dependency_reorder(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply dependency reordering adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Reorder dependencies due to: {failure_reason}]"
    )
    return adapted_plan


def _apply_step_modification(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply step modification adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Modify approach due to: {failure_reason}]"
    )
    return adapted_plan


def _apply_parallel_execution(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply parallel execution adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Execute in parallel due to: {failure_reason}]"
    )
    return adapted_plan


def _apply_fallback_strategy(original_plan: str, failed_step: str, failure_reason: str) -> str:
    """Apply fallback strategy adaptation."""
    adapted_plan = original_plan.replace(
        failed_step,
        f"{failed_step} [ADAPTED: Use fallback strategy due to: {failure_reason}]"
    )
    return adapted_plan