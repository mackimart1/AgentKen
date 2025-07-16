"""
Modular Agents Integration System
Demonstrates integration of Debugging, Learning, and Security agents with the core task execution pipeline
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent_framework import MessageBus, EnhancedOrchestrator, Message, MessageType
from agents.debugging_agent import DebuggingAgent
from agents.learning_agent import LearningAgent
from agents.security_agent import SecurityAgent


@dataclass
class SystemMetrics:
    """System-wide metrics for monitoring"""
    timestamp: datetime
    active_agents: int
    total_tasks_completed: int
    average_response_time: float
    error_rate: float
    security_events: int
    performance_score: float


class ModularAgentsOrchestrator:
    """
    Enhanced orchestrator that integrates the three modular agents
    with the core task execution pipeline
    """
    
    def __init__(self):
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Initialize core orchestrator
        self.core_orchestrator = EnhancedOrchestrator(self.message_bus)
        
        # Initialize modular agents
        self.debugging_agent = DebuggingAgent(self.message_bus)
        self.learning_agent = LearningAgent(self.message_bus)
        self.security_agent = SecurityAgent(self.message_bus)
        
        # System state
        self.system_metrics = []
        self.integration_active = True
        self.lifecycle_managers = {}
        
        # Initialize lifecycle management
        self._initialize_lifecycle_management()
        
        # Start integration monitoring
        self._start_integration_monitoring()
        
        logging.info("Modular Agents Orchestrator initialized with integrated pipeline")
    
    def _initialize_lifecycle_management(self):
        """Initialize lifecycle management for modular agents"""
        self.lifecycle_managers = {
            'debugging_agent': {
                'agent': self.debugging_agent,
                'health_check_interval': 60,
                'last_health_check': time.time(),
                'status': 'active',
                'restart_count': 0,
                'max_restarts': 3
            },
            'learning_agent': {
                'agent': self.learning_agent,
                'health_check_interval': 300,  # 5 minutes
                'last_health_check': time.time(),
                'status': 'active',
                'restart_count': 0,
                'max_restarts': 3
            },
            'security_agent': {
                'agent': self.security_agent,
                'health_check_interval': 30,
                'last_health_check': time.time(),
                'status': 'active',
                'restart_count': 0,
                'max_restarts': 3
            }
        }
    
    def _start_integration_monitoring(self):
        """Start background monitoring for integration health"""
        def monitoring_loop():
            while self.integration_active:
                try:
                    self._collect_system_metrics()
                    self._check_agent_health()
                    self._optimize_system_performance()
                    time.sleep(60)  # Monitor every minute
                except Exception as e:
                    logging.error(f"Integration monitoring error: {e}")
                    time.sleep(120)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # Get metrics from core orchestrator
            core_status = self.core_orchestrator.get_system_status()
            
            # Calculate performance metrics
            current_time = time.time()
            recent_metrics = [m for m in self.system_metrics 
                            if (current_time - m.timestamp.timestamp()) < 3600]  # Last hour
            
            avg_response_time = 0.0
            error_rate = 0.0
            if recent_metrics:
                avg_response_time = sum(m.average_response_time for m in recent_metrics) / len(recent_metrics)
                error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            
            # Get security events count
            security_events = len(self.security_agent.db.get_recent_events(hours=1))
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(avg_response_time, error_rate, security_events)
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                active_agents=core_status['agents'],
                total_tasks_completed=core_status['total_tasks_completed'],
                average_response_time=avg_response_time,
                error_rate=error_rate,
                security_events=security_events,
                performance_score=performance_score
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only last 24 hours of metrics
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_time]
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    def _calculate_performance_score(self, avg_response_time: float, error_rate: float, security_events: int) -> float:
        """Calculate overall system performance score"""
        # Base score
        score = 1.0
        
        # Penalize for slow response times (baseline: 5 seconds)
        if avg_response_time > 5.0:
            score -= min((avg_response_time - 5.0) / 10.0, 0.3)
        
        # Penalize for high error rates
        score -= min(error_rate, 0.4)
        
        # Penalize for security events
        if security_events > 0:
            score -= min(security_events * 0.05, 0.3)
        
        return max(0.0, score)
    
    def _check_agent_health(self):
        """Check health of modular agents"""
        current_time = time.time()
        
        for agent_name, manager in self.lifecycle_managers.items():
            try:
                # Check if health check is due
                if current_time - manager['last_health_check'] > manager['health_check_interval']:
                    health_status = self._perform_agent_health_check(agent_name, manager['agent'])
                    
                    if not health_status['healthy']:
                        logging.warning(f"Agent {agent_name} health check failed: {health_status['issues']}")
                        self._handle_unhealthy_agent(agent_name, manager, health_status)
                    else:
                        manager['status'] = 'active'
                        manager['restart_count'] = 0  # Reset restart count on successful health check
                    
                    manager['last_health_check'] = current_time
                    
            except Exception as e:
                logging.error(f"Error checking health of {agent_name}: {e}")
    
    def _perform_agent_health_check(self, agent_name: str, agent) -> Dict[str, Any]:
        """Perform health check on a specific agent"""
        health_status = {
            'healthy': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Check if agent is responsive
            if hasattr(agent, 'status') and agent.status.value != 'idle':
                if agent.status.value in ['error', 'offline']:
                    health_status['healthy'] = False
                    health_status['issues'].append(f"Agent status: {agent.status.value}")
            
            # Check active tasks
            if hasattr(agent, 'active_tasks'):
                active_count = len(agent.active_tasks)
                health_status['metrics']['active_tasks'] = active_count
                
                # Check for stuck tasks (running for more than 10 minutes)
                stuck_tasks = 0
                current_time = time.time()
                for task_id, task_info in agent.active_tasks.items():
                    if current_time - task_info.get('start_time', current_time) > 600:  # 10 minutes
                        stuck_tasks += 1
                
                if stuck_tasks > 0:
                    health_status['healthy'] = False
                    health_status['issues'].append(f"{stuck_tasks} stuck tasks detected")
            
            # Agent-specific health checks
            if agent_name == 'debugging_agent':
                health_status.update(self._check_debugging_agent_health(agent))
            elif agent_name == 'learning_agent':
                health_status.update(self._check_learning_agent_health(agent))
            elif agent_name == 'security_agent':
                health_status.update(self._check_security_agent_health(agent))
            
        except Exception as e:
            health_status['healthy'] = False
            health_status['issues'].append(f"Health check error: {str(e)}")
        
        return health_status
    
    def _check_debugging_agent_health(self, agent) -> Dict[str, Any]:
        """Specific health checks for debugging agent"""
        health_update = {'healthy': True, 'issues': []}
        
        try:
            # Check if monitoring is active
            if not agent.monitoring_active:
                health_update['healthy'] = False
                health_update['issues'].append("Monitoring is not active")
            
            # Check database connectivity
            try:
                agent.db.get_failure_patterns(limit=1)
            except Exception as e:
                health_update['healthy'] = False
                health_update['issues'].append(f"Database connectivity issue: {str(e)}")
            
        except Exception as e:
            health_update['issues'].append(f"Debugging agent health check error: {str(e)}")
        
        return health_update
    
    def _check_learning_agent_health(self, agent) -> Dict[str, Any]:
        """Specific health checks for learning agent"""
        health_update = {'healthy': True, 'issues': []}
        
        try:
            # Check if learning is active
            if not agent.learning_active:
                health_update['healthy'] = False
                health_update['issues'].append("Learning is not active")
            
            # Check database connectivity
            try:
                agent.db.get_performance_data(time_range_days=1)
            except Exception as e:
                health_update['healthy'] = False
                health_update['issues'].append(f"Database connectivity issue: {str(e)}")
            
        except Exception as e:
            health_update['issues'].append(f"Learning agent health check error: {str(e)}")
        
        return health_update
    
    def _check_security_agent_health(self, agent) -> Dict[str, Any]:
        """Specific health checks for security agent"""
        health_update = {'healthy': True, 'issues': []}
        
        try:
            # Check if monitoring is active
            if not agent.monitoring_active:
                health_update['healthy'] = False
                health_update['issues'].append("Security monitoring is not active")
            
            # Check for critical unmitigated events
            critical_events = agent.db.get_recent_events(hours=1, threat_level=agent.ThreatLevel.CRITICAL)
            unmitigated_critical = [e for e in critical_events if not e.mitigated]
            
            if len(unmitigated_critical) > 0:
                health_update['healthy'] = False
                health_update['issues'].append(f"{len(unmitigated_critical)} unmitigated critical security events")
            
        except Exception as e:
            health_update['issues'].append(f"Security agent health check error: {str(e)}")
        
        return health_update
    
    def _handle_unhealthy_agent(self, agent_name: str, manager: Dict[str, Any], health_status: Dict[str, Any]):
        """Handle an unhealthy agent"""
        manager['status'] = 'unhealthy'
        
        # Attempt restart if within limits
        if manager['restart_count'] < manager['max_restarts']:
            logging.info(f"Attempting to restart {agent_name} (attempt {manager['restart_count'] + 1})")
            
            try:
                # Send restart signal
                restart_message = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.SHUTDOWN,
                    sender="modular_orchestrator",
                    recipient=agent_name,
                    payload={'restart': True, 'reason': 'health_check_failure'},
                    timestamp=time.time()
                )
                
                self.message_bus.publish(restart_message)
                
                manager['restart_count'] += 1
                manager['last_health_check'] = time.time() + 60  # Wait 1 minute before next check
                
            except Exception as e:
                logging.error(f"Error restarting {agent_name}: {e}")
        else:
            logging.error(f"Agent {agent_name} has exceeded maximum restart attempts")
            manager['status'] = 'failed'
    
    def _optimize_system_performance(self):
        """Optimize system performance based on collected metrics"""
        try:
            if len(self.system_metrics) < 5:
                return  # Need sufficient data
            
            recent_metrics = self.system_metrics[-5:]  # Last 5 measurements
            avg_performance = sum(m.performance_score for m in recent_metrics) / len(recent_metrics)
            
            # If performance is degrading, trigger optimization
            if avg_performance < 0.7:
                logging.info("System performance degradation detected, triggering optimization")
                
                # Request learning agent to analyze performance
                optimization_request = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.TASK_REQUEST,
                    sender="modular_orchestrator",
                    recipient="learning_agent",
                    payload={
                        'task_type': 'optimization_recommendations',
                        'optimization_type': 'performance',
                        'urgency': 'high'
                    },
                    timestamp=time.time(),
                    priority=4
                )
                
                self.message_bus.publish(optimization_request)
            
            # Check for security concerns
            recent_security_events = sum(m.security_events for m in recent_metrics)
            if recent_security_events > 10:
                logging.warning("High security event rate detected")
                
                # Request security audit
                security_audit_request = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.TASK_REQUEST,
                    sender="modular_orchestrator",
                    recipient="security_agent",
                    payload={
                        'task_type': 'security_audit',
                        'audit_scope': 'system',
                        'urgency': 'high'
                    },
                    timestamp=time.time(),
                    priority=5
                )
                
                self.message_bus.publish(security_audit_request)
            
        except Exception as e:
            logging.error(f"Error optimizing system performance: {e}")
    
    def submit_task_with_monitoring(self, task_type: str, payload: Dict[str, Any], 
                                  priority: int = 1, timeout: Optional[float] = None) -> str:
        """Submit a task with integrated monitoring from modular agents"""
        try:
            # Security check before task submission
            security_check = self._perform_security_check(task_type, payload)
            if not security_check['approved']:
                raise SecurityError(f"Task rejected by security agent: {security_check['reason']}")
            
            # Submit task to core orchestrator
            task_id = self.core_orchestrator.submit_task(task_type, payload, priority, timeout)
            
            # Notify debugging agent of new task
            debug_notification = Message(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_REQUEST,
                sender="modular_orchestrator",
                recipient="debugging_agent",
                payload={
                    'task_type': 'monitor_task',
                    'monitored_task_id': task_id,
                    'task_details': {'type': task_type, 'priority': priority}
                },
                timestamp=time.time()
            )
            
            self.message_bus.publish(debug_notification)
            
            return task_id
            
        except Exception as e:
            # Report error to debugging agent
            error_report = Message(
                id=str(uuid.uuid4()),
                type=MessageType.ERROR_REPORT,
                sender="modular_orchestrator",
                recipient="debugging_agent",
                payload={
                    'error': str(e),
                    'context': {'task_type': task_type, 'payload_size': len(str(payload))}
                },
                timestamp=time.time()
            )
            
            self.message_bus.publish(error_report)
            raise
    
    def _perform_security_check(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security check on task submission"""
        try:
            # Basic security validation
            security_result = {
                'approved': True,
                'reason': '',
                'risk_level': 'low'
            }
            
            # Check payload size
            payload_str = json.dumps(payload)
            if len(payload_str) > 1048576:  # 1MB limit
                security_result['approved'] = False
                security_result['reason'] = 'Payload size exceeds security limit'
                security_result['risk_level'] = 'high'
                return security_result
            
            # Check for suspicious patterns
            threats = self.security_agent.threat_detector.detect_threats(payload_str, {
                'task_type': task_type,
                'source': 'task_submission'
            })
            
            if threats:
                high_severity_threats = [t for t in threats if t['severity'] in ['high', 'critical']]
                if high_severity_threats:
                    security_result['approved'] = False
                    security_result['reason'] = f"High severity threats detected: {[t['type'] for t in high_severity_threats]}"
                    security_result['risk_level'] = 'high'
            
            return security_result
            
        except Exception as e:
            logging.error(f"Error performing security check: {e}")
            return {'approved': False, 'reason': f'Security check error: {str(e)}', 'risk_level': 'unknown'}
    
    def get_integrated_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including modular agents"""
        try:
            # Get core system status
            core_status = self.core_orchestrator.get_system_status()
            
            # Get modular agents status
            modular_status = {}
            for agent_name, manager in self.lifecycle_managers.items():
                modular_status[agent_name] = {
                    'status': manager['status'],
                    'restart_count': manager['restart_count'],
                    'last_health_check': datetime.fromtimestamp(manager['last_health_check']).isoformat()
                }
            
            # Get recent metrics
            recent_metrics = self.system_metrics[-10:] if self.system_metrics else []
            
            # Calculate trends
            performance_trend = 'stable'
            if len(recent_metrics) >= 5:
                recent_scores = [m.performance_score for m in recent_metrics[-5:]]
                older_scores = [m.performance_score for m in recent_metrics[-10:-5]] if len(recent_metrics) >= 10 else recent_scores
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                if recent_avg > older_avg + 0.1:
                    performance_trend = 'improving'
                elif recent_avg < older_avg - 0.1:
                    performance_trend = 'degrading'
            
            return {
                'core_system': core_status,
                'modular_agents': modular_status,
                'system_metrics': {
                    'current_performance_score': recent_metrics[-1].performance_score if recent_metrics else 0.0,
                    'performance_trend': performance_trend,
                    'total_metrics_collected': len(self.system_metrics),
                    'integration_active': self.integration_active
                },
                'integration_health': {
                    'healthy_agents': len([m for m in self.lifecycle_managers.values() if m['status'] == 'active']),
                    'total_agents': len(self.lifecycle_managers),
                    'failed_agents': len([m for m in self.lifecycle_managers.values() if m['status'] == 'failed'])
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting integrated system status: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Gracefully shutdown the integrated system"""
        logging.info("Shutting down modular agents integration system")
        
        self.integration_active = False
        
        # Shutdown modular agents
        for agent_name, manager in self.lifecycle_managers.items():
            try:
                manager['agent'].shutdown()
                logging.info(f"Shutdown {agent_name}")
            except Exception as e:
                logging.error(f"Error shutting down {agent_name}: {e}")
        
        logging.info("Modular agents integration system shutdown complete")


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


# Integration API functions for external use
def create_integrated_system() -> ModularAgentsOrchestrator:
    """Create and initialize the integrated modular agents system"""
    return ModularAgentsOrchestrator()


def demonstrate_integration():
    """Demonstrate the integrated system functionality"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("MODULAR AGENTS INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create integrated system
    print("\n1. Initializing integrated system...")
    orchestrator = create_integrated_system()
    
    # Wait for initialization
    time.sleep(2)
    
    # Get system status
    print("\n2. Getting system status...")
    status = orchestrator.get_integrated_system_status()
    print(f"System Status: {json.dumps(status, indent=2, default=str)}")
    
    # Submit a test task with monitoring
    print("\n3. Submitting test task with integrated monitoring...")
    try:
        task_id = orchestrator.submit_task_with_monitoring(
            task_type="test_task",
            payload={"test_data": "integration_test", "complexity": 0.5},
            priority=2
        )
        print(f"Task submitted successfully: {task_id}")
    except Exception as e:
        print(f"Task submission failed: {e}")
    
    # Wait for some processing
    print("\n4. Waiting for system processing...")
    time.sleep(5)
    
    # Get updated status
    print("\n5. Getting updated system status...")
    updated_status = orchestrator.get_integrated_system_status()
    print(f"Updated Status: {json.dumps(updated_status, indent=2, default=str)}")
    
    # Test security functionality
    print("\n6. Testing security functionality...")
    try:
        # This should be rejected by security
        malicious_task_id = orchestrator.submit_task_with_monitoring(
            task_type="malicious_task",
            payload={"script": "DROP TABLE users; --", "command": "rm -rf /"},
            priority=1
        )
        print(f"Malicious task submitted: {malicious_task_id}")
    except SecurityError as e:
        print(f"Security correctly blocked malicious task: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Demonstrate learning capabilities
    print("\n7. Testing learning agent capabilities...")
    try:
        # Simulate some task data for learning
        learning_result = orchestrator.learning_agent.execute_capability("performance_analysis", {
            "time_range_days": 1,
            "analysis_type": "comprehensive"
        })
        print(f"Learning analysis completed: {learning_result.get('patterns', {}).keys()}")
    except Exception as e:
        print(f"Learning analysis error: {e}")
    
    # Demonstrate debugging capabilities
    print("\n8. Testing debugging agent capabilities...")
    try:
        debug_result = orchestrator.debugging_agent.execute_capability("system_health_check", {})
        print(f"System health check completed: {debug_result.get('health_status', {}).get('overall_status', 'unknown')}")
    except Exception as e:
        print(f"Debug health check error: {e}")
    
    # Final status
    print("\n9. Final system status...")
    final_status = orchestrator.get_integrated_system_status()
    print(f"Final Status Summary:")
    print(f"  - Core agents: {final_status.get('core_system', {}).get('agents', 0)}")
    print(f"  - Healthy modular agents: {final_status.get('integration_health', {}).get('healthy_agents', 0)}")
    print(f"  - Performance score: {final_status.get('system_metrics', {}).get('current_performance_score', 0.0):.2f}")
    print(f"  - Performance trend: {final_status.get('system_metrics', {}).get('performance_trend', 'unknown')}")
    
    # Cleanup
    print("\n10. Shutting down system...")
    orchestrator.shutdown()
    
    print("\nIntegration demonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_integration()