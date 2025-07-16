"""
Performance Monitoring System for AgentKen
Captures key metrics across agents and tools, provides dashboards and alerts.
"""

import asyncio
import json
import logging
import time
import statistics
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sqlite3
import numpy as np


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    SUCCESS_RATE = "success_rate"
    FAILURE_RATE = "failure_rate"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_LENGTH = "queue_length"
    ERROR_COUNT = "error_count"
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of system components"""
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"
    WORKFLOW = "workflow"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    id: str
    component_id: str
    component_type: ComponentType
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    component_id: str
    metric_type: MetricType
    level: AlertLevel
    title: str
    description: str
    threshold: float
    current_value: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class ComponentStats:
    """Statistics for a component"""
    component_id: str
    component_type: ComponentType
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_latency: float
    max_latency: float
    min_latency: float
    success_rate: float
    last_execution: Optional[float] = None
    avg_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0


class MetricsStorage:
    """Persistent storage for metrics"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    unit TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    resolution_time REAL
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component_id)")
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics 
                (id, component_id, component_type, metric_type, value, timestamp, unit, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.id,
                metric.component_id,
                metric.component_type.value,
                metric.metric_type.value,
                metric.value,
                metric.timestamp,
                metric.unit,
                json.dumps(metric.tags),
                json.dumps(metric.metadata)
            ))
    
    def store_alert(self, alert: PerformanceAlert):
        """Store a performance alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, component_id, metric_type, level, title, description, threshold, current_value, timestamp, acknowledged, resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.component_id,
                alert.metric_type.value,
                alert.level.value,
                alert.title,
                alert.description,
                alert.threshold,
                alert.current_value,
                alert.timestamp,
                int(alert.acknowledged),
                int(alert.resolved),
                alert.resolution_time
            ))
    
    def get_metrics(self, component_id: str = None, metric_type: MetricType = None, 
                   start_time: float = None, end_time: float = None, limit: int = 1000) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering"""
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if component_id:
            query += " AND component_id = ?"
            params.append(component_id)
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            metrics = []
            
            for row in cursor.fetchall():
                metric = PerformanceMetric(
                    id=row[0],
                    component_id=row[1],
                    component_type=ComponentType(row[2]),
                    metric_type=MetricType(row[3]),
                    value=row[4],
                    timestamp=row[5],
                    unit=row[6],
                    tags=json.loads(row[7]) if row[7] else {},
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                metrics.append(metric)
            
            return metrics
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active (unresolved) alerts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC")
            alerts = []
            
            for row in cursor.fetchall():
                alert = PerformanceAlert(
                    id=row[0],
                    component_id=row[1],
                    metric_type=MetricType(row[2]),
                    level=AlertLevel(row[3]),
                    title=row[4],
                    description=row[5],
                    threshold=row[6],
                    current_value=row[7],
                    timestamp=row[8],
                    acknowledged=bool(row[9]),
                    resolved=bool(row[10]),
                    resolution_time=row[11]
                )
                alerts.append(alert)
            
            return alerts


class PerformanceCollector:
    """Collects performance metrics from agents and tools"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_execution(self, component_id: str, component_type: ComponentType, 
                       operation: str, metadata: Dict[str, Any] = None) -> str:
        """Start tracking an execution"""
        execution_id = str(uuid.uuid4())
        
        with self._lock:
            self.active_executions[execution_id] = {
                'component_id': component_id,
                'component_type': component_type,
                'operation': operation,
                'start_time': time.time(),
                'metadata': metadata or {}
            }
        
        return execution_id
    
    def end_execution(self, execution_id: str, success: bool = True, 
                     error: str = None, result_metadata: Dict[str, Any] = None):
        """End tracking an execution and record metrics"""
        with self._lock:
            if execution_id not in self.active_executions:
                return
            
            execution = self.active_executions.pop(execution_id)
        
        end_time = time.time()
        duration = end_time - execution['start_time']
        
        # Record latency metric
        latency_metric = PerformanceMetric(
            id=str(uuid.uuid4()),
            component_id=execution['component_id'],
            component_type=execution['component_type'],
            metric_type=MetricType.LATENCY,
            value=duration * 1000,  # Convert to milliseconds
            timestamp=end_time,
            unit="ms",
            tags={
                'operation': execution['operation'],
                'success': str(success)
            },
            metadata={
                'execution_id': execution_id,
                'error': error,
                **(result_metadata or {})
            }
        )
        
        self.storage.store_metric(latency_metric)
        
        # Record success/failure metric
        success_metric = PerformanceMetric(
            id=str(uuid.uuid4()),
            component_id=execution['component_id'],
            component_type=execution['component_type'],
            metric_type=MetricType.SUCCESS_RATE if success else MetricType.FAILURE_RATE,
            value=1.0,
            timestamp=end_time,
            unit="count",
            tags={
                'operation': execution['operation']
            },
            metadata={
                'execution_id': execution_id,
                'error': error
            }
        )
        
        self.storage.store_metric(success_metric)
    
    def record_metric(self, component_id: str, component_type: ComponentType,
                     metric_type: MetricType, value: float, unit: str,
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a custom metric"""
        metric = PerformanceMetric(
            id=str(uuid.uuid4()),
            component_id=component_id,
            component_type=component_type,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.storage.store_metric(metric)
    
    def get_component_stats(self, component_id: str, 
                           time_window_hours: int = 24) -> ComponentStats:
        """Get statistics for a component"""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        metrics = self.storage.get_metrics(
            component_id=component_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return ComponentStats(
                component_id=component_id,
                component_type=ComponentType.SYSTEM,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                avg_latency=0.0,
                max_latency=0.0,
                min_latency=0.0,
                success_rate=0.0
            )
        
        # Separate metrics by type
        latency_metrics = [m for m in metrics if m.metric_type == MetricType.LATENCY]
        success_metrics = [m for m in metrics if m.metric_type == MetricType.SUCCESS_RATE]
        failure_metrics = [m for m in metrics if m.metric_type == MetricType.FAILURE_RATE]
        
        # Calculate statistics
        total_executions = len(success_metrics) + len(failure_metrics)
        successful_executions = len(success_metrics)
        failed_executions = len(failure_metrics)
        
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0
        
        if latency_metrics:
            latencies = [m.value for m in latency_metrics]
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0.0
        
        last_execution = max([m.timestamp for m in metrics]) if metrics else None
        
        return ComponentStats(
            component_id=component_id,
            component_type=metrics[0].component_type,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            avg_latency=avg_latency,
            max_latency=max_latency,
            min_latency=min_latency,
            success_rate=success_rate,
            last_execution=last_execution
        )


class AlertManager:
    """Manages performance alerts and thresholds"""
    
    def __init__(self, storage: MetricsStorage, collector: PerformanceCollector):
        self.storage = storage
        self.collector = collector
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable[[PerformanceAlert], None]] = []
        self._running = False
        self._alert_thread = None
    
    def start(self):
        """Start alert monitoring"""
        self._running = True
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()
        logging.info("Alert manager started")
    
    def stop(self):
        """Stop alert monitoring"""
        self._running = False
        if self._alert_thread:
            self._alert_thread.join()
        logging.info("Alert manager stopped")
    
    def add_alert_rule(self, component_id: str, metric_type: MetricType,
                      threshold: float, condition: str, level: AlertLevel,
                      description: str = ""):
        """Add an alert rule"""
        rule_id = f"{component_id}_{metric_type.value}_{condition}_{threshold}"
        
        self.alert_rules[rule_id] = {
            'component_id': component_id,
            'metric_type': metric_type,
            'threshold': threshold,
            'condition': condition,  # 'greater_than', 'less_than', 'equals'
            'level': level,
            'description': description,
            'enabled': True
        }
        
        logging.info(f"Added alert rule: {rule_id}")
    
    def add_notification_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        alerts = self.storage.get_active_alerts()
        for alert in alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.storage.store_alert(alert)
                break
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        alerts = self.storage.get_active_alerts()
        for alert in alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = time.time()
                self.storage.store_alert(alert)
                break
    
    def _alert_loop(self):
        """Main alert checking loop"""
        while self._running:
            try:
                self._check_alert_rules()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logging.error(f"Alert checking error: {e}")
    
    def _check_alert_rules(self):
        """Check all alert rules"""
        for rule_id, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
            
            try:
                self._evaluate_alert_rule(rule_id, rule)
            except Exception as e:
                logging.error(f"Alert rule evaluation failed for {rule_id}: {e}")
    
    def _evaluate_alert_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Evaluate a single alert rule"""
        component_id = rule['component_id']
        metric_type = rule['metric_type']
        threshold = rule['threshold']
        condition = rule['condition']
        
        # Get recent metrics (last 5 minutes)
        end_time = time.time()
        start_time = end_time - 300
        
        metrics = self.storage.get_metrics(
            component_id=component_id,
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return
        
        # Calculate current value (average of recent metrics)
        current_value = statistics.mean([m.value for m in metrics])
        
        # Evaluate condition
        triggered = False
        if condition == 'greater_than' and current_value > threshold:
            triggered = True
        elif condition == 'less_than' and current_value < threshold:
            triggered = True
        elif condition == 'equals' and abs(current_value - threshold) < 0.001:
            triggered = True
        
        # Check if alert already exists
        active_alerts = self.storage.get_active_alerts()
        existing_alert = next((a for a in active_alerts if a.id == rule_id), None)
        
        if triggered and not existing_alert:
            # Create new alert
            alert = PerformanceAlert(
                id=rule_id,
                component_id=component_id,
                metric_type=metric_type,
                level=rule['level'],
                title=f"Performance Alert: {component_id}",
                description=rule.get('description', f"{metric_type.value} {condition} {threshold}"),
                threshold=threshold,
                current_value=current_value,
                timestamp=time.time()
            )
            
            self.storage.store_alert(alert)
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logging.error(f"Notification handler failed: {e}")
        
        elif not triggered and existing_alert:
            # Auto-resolve alert
            self.resolve_alert(rule_id)


class PerformanceDashboard:
    """Generates performance dashboards and reports"""
    
    def __init__(self, storage: MetricsStorage, collector: PerformanceCollector):
        self.storage = storage
        self.collector = collector
    
    def generate_system_overview(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate system-wide performance overview"""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        # Get all metrics in time window
        all_metrics = self.storage.get_metrics(start_time=start_time, end_time=end_time)
        
        # Group by component
        components = defaultdict(list)
        for metric in all_metrics:
            components[metric.component_id].append(metric)
        
        # Calculate system-wide statistics
        total_executions = len([m for m in all_metrics if m.metric_type in [MetricType.SUCCESS_RATE, MetricType.FAILURE_RATE]])
        successful_executions = len([m for m in all_metrics if m.metric_type == MetricType.SUCCESS_RATE])
        failed_executions = len([m for m in all_metrics if m.metric_type == MetricType.FAILURE_RATE])
        
        system_success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0
        
        latency_metrics = [m for m in all_metrics if m.metric_type == MetricType.LATENCY]
        avg_system_latency = statistics.mean([m.value for m in latency_metrics]) if latency_metrics else 0.0
        
        # Get active alerts
        active_alerts = self.storage.get_active_alerts()
        
        # Component statistics
        component_stats = {}
        for component_id in components.keys():
            component_stats[component_id] = asdict(self.collector.get_component_stats(component_id, time_window_hours))
        
        return {
            'timestamp': time.time(),
            'time_window_hours': time_window_hours,
            'system_metrics': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'system_success_rate': system_success_rate,
                'avg_system_latency': avg_system_latency,
                'total_components': len(components),
                'active_alerts': len(active_alerts)
            },
            'component_stats': component_stats,
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'top_performers': self._get_top_performers(component_stats),
            'bottlenecks': self._identify_bottlenecks(component_stats)
        }
    
    def generate_component_report(self, component_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate detailed report for a specific component"""
        stats = self.collector.get_component_stats(component_id, time_window_hours)
        
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        # Get metrics for this component
        metrics = self.storage.get_metrics(
            component_id=component_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric)
        
        # Calculate trends
        trends = {}
        for metric_type, type_metrics in metrics_by_type.items():
            if len(type_metrics) >= 2:
                # Simple trend calculation (recent vs older)
                mid_point = len(type_metrics) // 2
                recent_avg = statistics.mean([m.value for m in type_metrics[:mid_point]])
                older_avg = statistics.mean([m.value for m in type_metrics[mid_point:]])
                
                trend = "improving" if recent_avg < older_avg else "degrading" if recent_avg > older_avg else "stable"
                trends[metric_type.value] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'change_percent': ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                }
        
        return {
            'component_id': component_id,
            'timestamp': time.time(),
            'time_window_hours': time_window_hours,
            'stats': asdict(stats),
            'trends': trends,
            'metrics_summary': {
                metric_type.value: {
                    'count': len(type_metrics),
                    'avg': statistics.mean([m.value for m in type_metrics]) if type_metrics else 0,
                    'min': min([m.value for m in type_metrics]) if type_metrics else 0,
                    'max': max([m.value for m in type_metrics]) if type_metrics else 0
                }
                for metric_type, type_metrics in metrics_by_type.items()
            }
        }
    
    def _get_top_performers(self, component_stats: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify top performing components"""
        performers = []
        
        for component_id, stats in component_stats.items():
            if stats['total_executions'] > 0:
                score = (stats['success_rate'] * 0.6) + ((1000 / max(stats['avg_latency'], 1)) * 0.4)
                performers.append({
                    'component_id': component_id,
                    'score': score,
                    'success_rate': stats['success_rate'],
                    'avg_latency': stats['avg_latency']
                })
        
        return sorted(performers, key=lambda x: x['score'], reverse=True)[:5]
    
    def _identify_bottlenecks(self, component_stats: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for component_id, stats in component_stats.items():
            issues = []
            
            if stats['success_rate'] < 90:
                issues.append(f"Low success rate: {stats['success_rate']:.1f}%")
            
            if stats['avg_latency'] > 5000:  # > 5 seconds
                issues.append(f"High latency: {stats['avg_latency']:.0f}ms")
            
            if issues:
                bottlenecks.append({
                    'component_id': component_id,
                    'issues': issues,
                    'severity': len(issues)
                })
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)


class PerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.storage = MetricsStorage(db_path)
        self.collector = PerformanceCollector(self.storage)
        self.alert_manager = AlertManager(self.storage, self.collector)
        self.dashboard = PerformanceDashboard(self.storage, self.collector)
        
        self._setup_default_alerts()
    
    def start(self):
        """Start the performance monitoring system"""
        self.alert_manager.start()
        logging.info("Performance monitoring system started")
    
    def stop(self):
        """Stop the performance monitoring system"""
        self.alert_manager.stop()
        logging.info("Performance monitoring system stopped")
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High latency alerts
        self.alert_manager.add_alert_rule(
            component_id="*",  # Apply to all components
            metric_type=MetricType.LATENCY,
            threshold=5000.0,  # 5 seconds
            condition="greater_than",
            level=AlertLevel.WARNING,
            description="Component latency is high"
        )
        
        # Low success rate alerts
        self.alert_manager.add_alert_rule(
            component_id="*",
            metric_type=MetricType.SUCCESS_RATE,
            threshold=90.0,
            condition="less_than",
            level=AlertLevel.ERROR,
            description="Component success rate is low"
        )
    
    def track_execution(self, component_id: str, component_type: ComponentType, operation: str):
        """Context manager for tracking executions"""
        return ExecutionTracker(self.collector, component_id, component_type, operation)
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        overview = self.dashboard.generate_system_overview(time_window_hours=1)
        
        # Start with perfect score
        health_score = 100.0
        
        # Deduct for low success rate
        success_rate = overview['system_metrics']['system_success_rate']
        if success_rate < 95:
            health_score -= (95 - success_rate) * 2
        
        # Deduct for high latency
        avg_latency = overview['system_metrics']['avg_system_latency']
        if avg_latency > 1000:  # > 1 second
            health_score -= min((avg_latency - 1000) / 100, 30)
        
        # Deduct for active alerts
        active_alerts = len(overview['active_alerts'])
        health_score -= min(active_alerts * 5, 40)
        
        return max(0.0, min(100.0, health_score))


class ExecutionTracker:
    """Context manager for tracking execution performance"""
    
    def __init__(self, collector: PerformanceCollector, component_id: str, 
                 component_type: ComponentType, operation: str):
        self.collector = collector
        self.component_id = component_id
        self.component_type = component_type
        self.operation = operation
        self.execution_id = None
    
    def __enter__(self):
        self.execution_id = self.collector.start_execution(
            self.component_id, self.component_type, self.operation
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        
        self.collector.end_execution(
            self.execution_id, success=success, error=error
        )


# Notification handlers
def console_notification_handler(alert: PerformanceAlert):
    """Console notification handler"""
    print(f"\n🚨 PERFORMANCE ALERT [{alert.level.value.upper()}]")
    print(f"Component: {alert.component_id}")
    print(f"Metric: {alert.metric_type.value}")
    print(f"Current: {alert.current_value:.2f}, Threshold: {alert.threshold:.2f}")
    print(f"Description: {alert.description}")
    print(f"Time: {datetime.fromtimestamp(alert.timestamp)}")


def log_notification_handler(alert: PerformanceAlert):
    """Log-based notification handler"""
    logging.warning(
        f"Performance alert: {alert.component_id} - {alert.metric_type.value} "
        f"({alert.current_value:.2f} vs {alert.threshold:.2f})"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Add notification handlers
    monitor.alert_manager.add_notification_handler(console_notification_handler)
    monitor.alert_manager.add_notification_handler(log_notification_handler)
    
    # Start monitoring
    monitor.start()
    
    # Simulate some agent/tool executions
    print("Simulating agent and tool executions...")
    
    # Simulate successful execution
    with monitor.track_execution("research_agent", ComponentType.AGENT, "web_search"):
        time.sleep(0.1)  # Simulate work
    
    # Simulate tool execution with higher latency
    with monitor.track_execution("web_scraper_tool", ComponentType.TOOL, "scrape_page"):
        time.sleep(0.5)  # Simulate slower work
    
    # Simulate failed execution
    try:
        with monitor.track_execution("data_processor", ComponentType.TOOL, "process_data"):
            time.sleep(0.2)
            raise Exception("Processing failed")
    except:
        pass
    
    # Wait for metrics to be processed
    time.sleep(2)
    
    # Generate reports
    print("\n=== SYSTEM OVERVIEW ===")
    overview = monitor.dashboard.generate_system_overview()
    print(f"System Health Score: {monitor.get_system_health_score():.1f}/100")
    print(f"Total Executions: {overview['system_metrics']['total_executions']}")
    print(f"Success Rate: {overview['system_metrics']['system_success_rate']:.1f}%")
    print(f"Average Latency: {overview['system_metrics']['avg_system_latency']:.1f}ms")
    print(f"Active Alerts: {overview['system_metrics']['active_alerts']}")
    
    if overview['bottlenecks']:
        print("\n=== BOTTLENECKS ===")
        for bottleneck in overview['bottlenecks']:
            print(f"- {bottleneck['component_id']}: {', '.join(bottleneck['issues'])}")
    
    # Stop monitoring
    monitor.stop()