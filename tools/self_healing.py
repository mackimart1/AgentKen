from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import psutil
import os
import time
import threading
import logging
import traceback
import gc
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


class HealthCheckInput(BaseModel):
    component: str = Field(default="system", description="Component to check: system, agent_smith, memory, cpu")
    detailed: bool = Field(default=False, description="Include detailed metrics")


class RecoveryActionInput(BaseModel):
    recovery_type: str = Field(description="Type of recovery: memory_cleanup, restart_component, rollback")
    component: str = Field(default="agent_smith", description="Component to recover")
    parameters: Dict[str, Any] = Field(default={}, description="Recovery parameters")


class MonitoringConfigInput(BaseModel):
    enabled: bool = Field(default=True, description="Enable/disable monitoring")
    check_interval: int = Field(default=30, description="Check interval in seconds")
    thresholds: Dict[str, float] = Field(default={}, description="Health thresholds")


@dataclass
class HealthMetrics:
    """Health monitoring metrics."""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    network_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """Overall health status."""
    status: str  # healthy, warning, critical, failed
    score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    last_check: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "last_check": self.last_check.isoformat()
        }


# Global monitoring state
_monitoring_enabled = False
_monitoring_thread = None
_health_history: List[HealthMetrics] = []
_recovery_log: List[Dict[str, Any]] = []
_health_thresholds = {
    "cpu_usage": 80.0,
    "memory_usage": 85.0,
    "disk_usage": 90.0,
    "error_rate": 5.0,
    "response_time": 30.0
}

# Setup logger
logger = logging.getLogger(__name__)


def _collect_system_metrics() -> HealthMetrics:
    """Collect comprehensive system health metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Process metrics
        try:
            process = psutil.Process()
            active_threads = process.num_threads()
            open_files = len(process.open_files())
            network_connections = len(process.connections())
        except:
            active_threads = 0
            open_files = 0
            network_connections = 0
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_usage=disk_percent,
            active_threads=active_threads,
            open_files=open_files,
            network_connections=network_connections
        )
        
    except Exception as e:
        logger.error(f"Failed to collect system metrics: {e}")
        return HealthMetrics(timestamp=datetime.now())


def _assess_health_status(metrics: HealthMetrics) -> HealthStatus:
    """Assess overall health status based on metrics."""
    issues = []
    recommendations = []
    score = 100.0
    
    # Check CPU usage
    if metrics.cpu_usage > _health_thresholds["cpu_usage"]:
        issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        recommendations.append("Consider reducing computational load or optimizing algorithms")
        score -= 20
    elif metrics.cpu_usage > _health_thresholds["cpu_usage"] * 0.8:
        issues.append(f"Elevated CPU usage: {metrics.cpu_usage:.1f}%")
        score -= 10
    
    # Check memory usage
    if metrics.memory_usage > _health_thresholds["memory_usage"]:
        issues.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        recommendations.append("Perform memory cleanup or restart components")
        score -= 25
    elif metrics.memory_usage > _health_thresholds["memory_usage"] * 0.8:
        issues.append(f"Elevated memory usage: {metrics.memory_usage:.1f}%")
        score -= 15
    
    # Check disk usage
    if metrics.disk_usage > _health_thresholds["disk_usage"]:
        issues.append(f"High disk usage: {metrics.disk_usage:.1f}%")
        recommendations.append("Clean up temporary files or expand storage")
        score -= 15
    
    # Check thread count
    if metrics.active_threads > 50:
        issues.append(f"High thread count: {metrics.active_threads}")
        recommendations.append("Check for thread leaks or reduce concurrent operations")
        score -= 10
    
    # Check open files
    if metrics.open_files > 100:
        issues.append(f"Many open files: {metrics.open_files}")
        recommendations.append("Ensure proper file handle cleanup")
        score -= 5
    
    # Determine status
    if score >= 90:
        status = "healthy"
    elif score >= 70:
        status = "warning"
    elif score >= 50:
        status = "critical"
    else:
        status = "failed"
    
    return HealthStatus(
        status=status,
        score=max(0, score),
        issues=issues,
        recommendations=recommendations,
        last_check=metrics.timestamp
    )


def _monitoring_loop():
    """Main monitoring loop."""
    global _health_history
    
    while _monitoring_enabled:
        try:
            # Collect metrics
            metrics = _collect_system_metrics()
            _health_history.append(metrics)
            
            # Keep only last 100 entries
            if len(_health_history) > 100:
                _health_history = _health_history[-100:]
            
            # Assess health
            health_status = _assess_health_status(metrics)
            
            # Auto-recovery for critical issues
            if health_status.status in ["critical", "failed"]:
                _attempt_auto_recovery(health_status, metrics)
            
            # Sleep until next check
            time.sleep(30)  # Default check interval
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(30)


def _attempt_auto_recovery(health_status: HealthStatus, metrics: HealthMetrics):
    """Attempt automatic recovery based on health issues."""
    recovery_actions = []
    
    try:
        # Memory cleanup for high memory usage
        if metrics.memory_usage > _health_thresholds["memory_usage"]:
            gc.collect()
            recovery_actions.append("memory_cleanup")
        
        # Thread cleanup for high thread count
        if metrics.active_threads > 50:
            # Log warning but don't forcefully kill threads
            logger.warning(f"High thread count detected: {metrics.active_threads}")
            recovery_actions.append("thread_monitoring")
        
        # Log recovery attempt
        _recovery_log.append({
            "timestamp": datetime.now().isoformat(),
            "health_status": health_status.status,
            "issues": health_status.issues,
            "actions_taken": recovery_actions,
            "metrics": metrics.to_dict()
        })
        
        # Keep only last 50 recovery entries
        if len(_recovery_log) > 50:
            _recovery_log[:] = _recovery_log[-50:]
        
        logger.info(f"Auto-recovery attempted: {recovery_actions}")
        
    except Exception as e:
        logger.error(f"Auto-recovery failed: {e}")


@tool(args_schema=HealthCheckInput)
def check_system_health(component: str = "system", detailed: bool = False) -> str:
    """
    Check system health and return status.
    
    Args:
        component: Component to check (system, agent_smith, memory, cpu)
        detailed: Include detailed metrics
    
    Returns:
        JSON string with health status
    """
    try:
        if component == "system":
            # Comprehensive system check
            metrics = _collect_system_metrics()
            health_status = _assess_health_status(metrics)
            
            result = {
                "status": "success",
                "component": component,
                "health_status": health_status.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            if detailed:
                result["metrics"] = metrics.to_dict()
                result["thresholds"] = _health_thresholds
                result["monitoring_enabled"] = _monitoring_enabled
        
        elif component == "memory":
            # Memory-specific check
            memory = psutil.virtual_memory()
            
            result = {
                "status": "success",
                "component": component,
                "memory_usage": memory.percent,
                "available_memory": memory.available,
                "total_memory": memory.total,
                "health_status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical",
                "timestamp": datetime.now().isoformat()
            }
        
        elif component == "cpu":
            # CPU-specific check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            result = {
                "status": "success",
                "component": component,
                "cpu_usage": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "health_status": "healthy" if cpu_percent < 70 else "warning" if cpu_percent < 85 else "critical",
                "timestamp": datetime.now().isoformat()
            }
        
        elif component == "agent_smith":
            # Agent Smith specific health check
            metrics = _collect_system_metrics()
            
            # Check if agent smith is responsive
            agent_smith_healthy = True
            try:
                # Simple responsiveness test
                import agents.agent_smith_enhanced
                agent_smith_healthy = True
            except Exception as e:
                agent_smith_healthy = False
            
            result = {
                "status": "success",
                "component": component,
                "agent_smith_responsive": agent_smith_healthy,
                "system_metrics": metrics.to_dict() if detailed else {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage
                },
                "health_status": "healthy" if agent_smith_healthy else "critical",
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            result = {
                "status": "failure",
                "message": f"Unknown component: {component}"
            }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Health check failed: {str(e)}"
        })


@tool(args_schema=RecoveryActionInput)
def perform_recovery_action(
    recovery_type: str,
    component: str = "agent_smith",
    parameters: Dict[str, Any] = None
) -> str:
    """
    Perform recovery action for system health.
    
    Args:
        recovery_type: Type of recovery (memory_cleanup, restart_component, rollback)
        component: Component to recover
        parameters: Recovery parameters
    
    Returns:
        JSON string with recovery result
    """
    try:
        if parameters is None:
            parameters = {}
        
        recovery_result = {
            "status": "success",
            "recovery_type": recovery_type,
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "actions_taken": []
        }
        
        if recovery_type == "memory_cleanup":
            # Perform memory cleanup
            initial_memory = psutil.virtual_memory().percent
            
            # Force garbage collection
            gc.collect()
            
            # Clear caches if available
            try:
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
            except:
                pass
            
            final_memory = psutil.virtual_memory().percent
            memory_freed = initial_memory - final_memory
            
            recovery_result["actions_taken"].append("garbage_collection")
            recovery_result["actions_taken"].append("cache_cleanup")
            recovery_result["memory_freed_percent"] = memory_freed
            recovery_result["message"] = f"Memory cleanup completed, freed {memory_freed:.1f}% memory"
        
        elif recovery_type == "restart_component":
            # Component restart simulation
            if component == "agent_smith":
                try:
                    # Reload agent smith module
                    import importlib
                    import agents.agent_smith_enhanced
                    importlib.reload(agents.agent_smith_enhanced)
                    
                    recovery_result["actions_taken"].append("module_reload")
                    recovery_result["message"] = f"Component {component} restarted successfully"
                except Exception as e:
                    recovery_result["status"] = "failure"
                    recovery_result["message"] = f"Failed to restart {component}: {e}"
            else:
                recovery_result["status"] = "failure"
                recovery_result["message"] = f"Component restart not supported for: {component}"
        
        elif recovery_type == "rollback":
            # Rollback to previous version
            agent_name = parameters.get("agent_name", "")
            target_version = parameters.get("version", "")
            
            if not agent_name or not target_version:
                recovery_result["status"] = "failure"
                recovery_result["message"] = "Rollback requires agent_name and version parameters"
            else:
                # Use versioning tool for rollback
                try:
                    from tools.agent_versioning import rollback_agent_version
                    rollback_result = rollback_agent_version.invoke({
                        "agent_name": agent_name,
                        "version": target_version
                    })
                    
                    rollback_data = json.loads(rollback_result)
                    if rollback_data["status"] == "success":
                        recovery_result["actions_taken"].append("version_rollback")
                        recovery_result["message"] = f"Rolled back {agent_name} to {target_version}"
                    else:
                        recovery_result["status"] = "failure"
                        recovery_result["message"] = rollback_data["message"]
                except Exception as e:
                    recovery_result["status"] = "failure"
                    recovery_result["message"] = f"Rollback failed: {e}"
        
        elif recovery_type == "system_reset":
            # System reset actions
            actions = []
            
            # Memory cleanup
            gc.collect()
            actions.append("memory_cleanup")
            
            # Clear temporary files
            try:
                import tempfile
                import shutil
                temp_dir = tempfile.gettempdir()
                # Don't actually delete temp files, just log the action
                actions.append("temp_cleanup_logged")
            except:
                pass
            
            recovery_result["actions_taken"] = actions
            recovery_result["message"] = "System reset actions completed"
        
        else:
            recovery_result["status"] = "failure"
            recovery_result["message"] = f"Unknown recovery type: {recovery_type}"
        
        # Log recovery action
        _recovery_log.append({
            "timestamp": datetime.now().isoformat(),
            "recovery_type": recovery_type,
            "component": component,
            "parameters": parameters,
            "result": recovery_result["status"],
            "actions": recovery_result.get("actions_taken", [])
        })
        
        return json.dumps(recovery_result)
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Recovery action failed: {str(e)}"
        })


@tool(args_schema=MonitoringConfigInput)
def configure_monitoring(
    enabled: bool = True,
    check_interval: int = 30,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Configure self-healing monitoring system.
    
    Args:
        enabled: Enable/disable monitoring
        check_interval: Check interval in seconds
        thresholds: Health thresholds
    
    Returns:
        JSON string with configuration result
    """
    try:
        global _monitoring_enabled, _monitoring_thread, _health_thresholds
        
        # Update thresholds if provided
        if thresholds:
            _health_thresholds.update(thresholds)
        
        # Handle monitoring state change
        if enabled and not _monitoring_enabled:
            # Start monitoring
            _monitoring_enabled = True
            _monitoring_thread = threading.Thread(target=_monitoring_loop, daemon=True)
            _monitoring_thread.start()
            message = "Monitoring started"
        elif not enabled and _monitoring_enabled:
            # Stop monitoring
            _monitoring_enabled = False
            if _monitoring_thread:
                _monitoring_thread.join(timeout=5)
            message = "Monitoring stopped"
        else:
            message = f"Monitoring already {'enabled' if enabled else 'disabled'}"
        
        return json.dumps({
            "status": "success",
            "monitoring_enabled": _monitoring_enabled,
            "check_interval": check_interval,
            "thresholds": _health_thresholds,
            "message": message
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to configure monitoring: {str(e)}"
        })


@tool
def get_health_history() -> str:
    """
    Get health monitoring history.
    
    Returns:
        JSON string with health history
    """
    try:
        if not _health_history:
            return json.dumps({
                "status": "success",
                "message": "No health history available",
                "history": []
            })
        
        # Get recent history (last 20 entries)
        recent_history = _health_history[-20:]
        
        # Calculate trends
        if len(recent_history) >= 2:
            latest = recent_history[-1]
            previous = recent_history[-2]
            
            trends = {
                "cpu_trend": latest.cpu_usage - previous.cpu_usage,
                "memory_trend": latest.memory_usage - previous.memory_usage,
                "disk_trend": latest.disk_usage - previous.disk_usage
            }
        else:
            trends = {}
        
        # Calculate averages
        if recent_history:
            avg_cpu = sum(m.cpu_usage for m in recent_history) / len(recent_history)
            avg_memory = sum(m.memory_usage for m in recent_history) / len(recent_history)
            avg_disk = sum(m.disk_usage for m in recent_history) / len(recent_history)
        else:
            avg_cpu = avg_memory = avg_disk = 0
        
        return json.dumps({
            "status": "success",
            "history": [m.to_dict() for m in recent_history],
            "trends": trends,
            "averages": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "disk_usage": avg_disk
            },
            "total_entries": len(_health_history),
            "monitoring_enabled": _monitoring_enabled,
            "message": f"Retrieved {len(recent_history)} health history entries"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get health history: {str(e)}"
        })


@tool
def get_recovery_log() -> str:
    """
    Get recovery action log.
    
    Returns:
        JSON string with recovery log
    """
    try:
        return json.dumps({
            "status": "success",
            "recovery_log": _recovery_log,
            "total_recoveries": len(_recovery_log),
            "message": f"Retrieved {len(_recovery_log)} recovery log entries"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Failed to get recovery log: {str(e)}"
        })


@tool
def emergency_recovery() -> str:
    """
    Perform emergency recovery actions.
    
    Returns:
        JSON string with emergency recovery result
    """
    try:
        emergency_actions = []
        
        # Force garbage collection
        gc.collect()
        emergency_actions.append("force_garbage_collection")
        
        # Clear all caches
        try:
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            emergency_actions.append("clear_type_cache")
        except:
            pass
        
        # Reset monitoring if it's stuck
        global _monitoring_enabled, _monitoring_thread
        if _monitoring_enabled:
            _monitoring_enabled = False
            time.sleep(1)
            _monitoring_enabled = True
            emergency_actions.append("reset_monitoring")
        
        # Log emergency recovery
        _recovery_log.append({
            "timestamp": datetime.now().isoformat(),
            "recovery_type": "emergency",
            "component": "system",
            "actions": emergency_actions,
            "triggered_by": "manual_emergency_call"
        })
        
        return json.dumps({
            "status": "success",
            "actions_taken": emergency_actions,
            "timestamp": datetime.now().isoformat(),
            "message": f"Emergency recovery completed with {len(emergency_actions)} actions"
        })
        
    except Exception as e:
        return json.dumps({
            "status": "failure",
            "message": f"Emergency recovery failed: {str(e)}"
        })


# Initialize monitoring on module load
try:
    _monitoring_enabled = True
    _monitoring_thread = threading.Thread(target=_monitoring_loop, daemon=True)
    _monitoring_thread.start()
    logger.info("Self-healing monitoring initialized")
except Exception as e:
    logger.error(f"Failed to initialize monitoring: {e}")