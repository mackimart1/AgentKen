"""
Workflow Optimization and Monitoring System
Provides comprehensive performance monitoring, bottleneck detection, and optimization recommendations.
"""

import asyncio
import json
import logging
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta


class MetricType(Enum):
    """Types of performance metrics"""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    QUEUE_LENGTH = "queue_length"
    SUCCESS_RATE = "success_rate"
    COST = "cost"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OptimizationStrategy(Enum):
    """Optimization strategies"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    LOAD_BALANCE = "load_balance"
    CACHE_OPTIMIZATION = "cache_optimization"
    RESOURCE_REALLOCATION = "resource_reallocation"
    WORKFLOW_RESTRUCTURE = "workflow_restructure"


@dataclass
class Metric:
    """Individual performance metric"""

    name: str
    type: MetricType
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert"""

    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""

    metric_name: str
    baseline_value: float
    acceptable_deviation: float
    measurement_window: int  # Number of measurements to consider
    last_updated: float


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""

    id: str
    strategy: OptimizationStrategy
    title: str
    description: str
    expected_improvement: float
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    priority_score: float
    affected_components: List[str]
    timestamp: float


@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis result"""

    component_name: str
    bottleneck_type: str
    severity_score: float
    impact_description: str
    suggested_solutions: List[str]
    measurement_data: Dict[str, Any]


class MetricsCollector:
    """Collects and stores performance metrics"""

    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_seconds = retention_hours * 3600
        self.collection_interval = 1.0  # seconds
        self.collectors: List[Callable] = []

        self._running = False
        self._collector_thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start metrics collection"""
        self._running = True
        self._collector_thread = threading.Thread(
            target=self._collect_loop, daemon=True
        )
        self._collector_thread.start()
        logging.info("Metrics collector started")

    def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join()
        logging.info("Metrics collector stopped")

    def add_collector(self, collector: Callable[[], List[Metric]]):
        """Add a metrics collector function"""
        with self._lock:
            self.collectors.append(collector)

    def record_metric(self, metric: Metric):
        """Record a single metric"""
        with self._lock:
            self.metrics[metric.name].append(metric)
            self._cleanup_old_metrics(metric.name)

    def record_metrics(self, metrics: List[Metric]):
        """Record multiple metrics"""
        for metric in metrics:
            self.record_metric(metric)

    def get_metrics(
        self,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Metric]:
        """Get metrics for a specific name within time range"""
        with self._lock:
            if metric_name not in self.metrics:
                return []

            metrics = list(self.metrics[metric_name])

            # Filter by time range
            if start_time is not None:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time is not None:
                metrics = [m for m in metrics if m.timestamp <= end_time]

            return metrics

    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest metric value"""
        with self._lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            return self.metrics[metric_name][-1]

    def get_metric_statistics(
        self, metric_name: str, window_seconds: int = 300
    ) -> Dict[str, float]:
        """Get statistical summary of metrics"""
        end_time = time.time()
        start_time = end_time - window_seconds

        metrics = self.get_metrics(metric_name, start_time, end_time)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "percentile_95": np.percentile(values, 95) if values else 0.0,
            "percentile_99": np.percentile(values, 99) if values else 0.0,
        }

    def _collect_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                # Collect from all registered collectors
                for collector in self.collectors:
                    try:
                        metrics = collector()
                        self.record_metrics(metrics)
                    except Exception as e:
                        logging.error(f"Metrics collector failed: {e}")

                time.sleep(self.collection_interval)

            except Exception as e:
                logging.error(f"Metrics collection loop error: {e}")

    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period"""
        cutoff_time = time.time() - self.retention_seconds
        metrics_deque = self.metrics[metric_name]

        # Remove old metrics from the front
        while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
            metrics_deque.popleft()


class AlertManager:
    """Manages performance alerts and notifications"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []

        self._running = False
        self._alert_thread = None
        self._lock = threading.Lock()

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

    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        condition: str,
        severity: AlertSeverity,
        description: str = "",
        window_seconds: int = 60,
    ):
        """Add an alert rule"""
        rule_id = f"{metric_name}_{condition}_{threshold}"

        with self._lock:
            self.alert_rules[rule_id] = {
                "metric_name": metric_name,
                "threshold": threshold,
                "condition": condition,  # 'greater_than', 'less_than', 'equals'
                "severity": severity,
                "description": description,
                "window_seconds": window_seconds,
                "enabled": True,
            }

        logging.info(f"Added alert rule: {rule_id}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        with self._lock:
            self.notification_handlers.append(handler)

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logging.info(f"Alert {alert_id} acknowledged")

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = time.time()

                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]

                logging.info(f"Alert {alert_id} resolved")

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        with self._lock:
            alerts = list(self.active_alerts.values())

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def _alert_loop(self):
        """Main alert checking loop"""
        while self._running:
            try:
                self._check_alert_rules()
                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logging.error(f"Alert checking error: {e}")

    def _check_alert_rules(self):
        """Check all alert rules"""
        with self._lock:
            for rule_id, rule in self.alert_rules.items():
                if not rule["enabled"]:
                    continue

                try:
                    self._evaluate_alert_rule(rule_id, rule)
                except Exception as e:
                    logging.error(f"Alert rule evaluation failed for {rule_id}: {e}")

    def _evaluate_alert_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Evaluate a single alert rule"""
        metric_name = rule["metric_name"]
        threshold = rule["threshold"]
        condition = rule["condition"]

        # Get recent metrics
        window_seconds = rule.get("window_seconds", 60)
        stats = self.metrics_collector.get_metric_statistics(
            metric_name, window_seconds
        )

        if not stats:
            return

        # Evaluate condition against mean value
        current_value = stats["mean"]
        triggered = False

        if condition == "greater_than" and current_value > threshold:
            triggered = True
        elif condition == "less_than" and current_value < threshold:
            triggered = True
        elif condition == "equals" and abs(current_value - threshold) < 0.001:
            triggered = True

        # Create or resolve alert
        if triggered and rule_id not in self.active_alerts:
            self._create_alert(rule_id, rule, current_value)
        elif not triggered and rule_id in self.active_alerts:
            self.resolve_alert(rule_id)

    def _create_alert(self, rule_id: str, rule: Dict[str, Any], current_value: float):
        """Create a new alert"""
        alert = Alert(
            id=rule_id,
            severity=rule["severity"],
            title=f"Alert: {rule['metric_name']}",
            description=rule.get(
                "description",
                f"Metric {rule['metric_name']} {rule['condition']} {rule['threshold']}",
            ),
            metric_name=rule["metric_name"],
            threshold=rule["threshold"],
            current_value=current_value,
            timestamp=time.time(),
        )

        self.active_alerts[rule_id] = alert

        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Notification handler failed: {e}")

        logging.warning(f"Alert created: {alert.title}")


class BottleneckDetector:
    """Detects performance bottlenecks in the system"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.detection_algorithms = [
            self._detect_latency_bottlenecks,
            self._detect_throughput_bottlenecks,
            self._detect_resource_bottlenecks,
            self._detect_queue_bottlenecks,
        ]

    def analyze_bottlenecks(
        self, window_seconds: int = 300
    ) -> List[BottleneckAnalysis]:
        """Analyze system for bottlenecks"""
        bottlenecks = []

        for algorithm in self.detection_algorithms:
            try:
                detected = algorithm(window_seconds)
                bottlenecks.extend(detected)
            except Exception as e:
                logging.error(f"Bottleneck detection algorithm failed: {e}")

        # Sort by severity score
        bottlenecks.sort(key=lambda b: b.severity_score, reverse=True)

        return bottlenecks

    def _detect_latency_bottlenecks(
        self, window_seconds: int
    ) -> List[BottleneckAnalysis]:
        """Detect latency-based bottlenecks"""
        bottlenecks = []

        # Look for components with high latency
        latency_metrics = [
            name
            for name in self.metrics_collector.metrics.keys()
            if "latency" in name.lower()
        ]

        for metric_name in latency_metrics:
            stats = self.metrics_collector.get_metric_statistics(
                metric_name, window_seconds
            )

            if stats and stats["percentile_95"] > 1000:  # > 1 second
                severity = min(stats["percentile_95"] / 1000, 10.0)  # Scale to 0-10

                bottleneck = BottleneckAnalysis(
                    component_name=metric_name.replace("_latency", ""),
                    bottleneck_type="latency",
                    severity_score=severity,
                    impact_description=f"High latency detected: P95 = {stats['percentile_95']:.2f}ms",
                    suggested_solutions=[
                        "Optimize algorithm efficiency",
                        "Add caching layer",
                        "Scale horizontally",
                        "Optimize database queries",
                    ],
                    measurement_data=stats,
                )
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _detect_throughput_bottlenecks(
        self, window_seconds: int
    ) -> List[BottleneckAnalysis]:
        """Detect throughput-based bottlenecks"""
        bottlenecks = []

        # Look for components with declining throughput
        throughput_metrics = [
            name
            for name in self.metrics_collector.metrics.keys()
            if "throughput" in name.lower() or "requests_per_second" in name.lower()
        ]

        for metric_name in throughput_metrics:
            stats = self.metrics_collector.get_metric_statistics(
                metric_name, window_seconds
            )

            if stats and stats["mean"] < 10:  # Less than 10 requests/second
                severity = 10.0 - stats["mean"]  # Higher severity for lower throughput

                bottleneck = BottleneckAnalysis(
                    component_name=metric_name.replace("_throughput", ""),
                    bottleneck_type="throughput",
                    severity_score=severity,
                    impact_description=f"Low throughput detected: {stats['mean']:.2f} req/s",
                    suggested_solutions=[
                        "Scale up resources",
                        "Optimize processing logic",
                        "Implement load balancing",
                        "Remove processing bottlenecks",
                    ],
                    measurement_data=stats,
                )
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _detect_resource_bottlenecks(
        self, window_seconds: int
    ) -> List[BottleneckAnalysis]:
        """Detect resource utilization bottlenecks"""
        bottlenecks = []

        # Look for high resource utilization
        resource_metrics = [
            name
            for name in self.metrics_collector.metrics.keys()
            if any(res in name.lower() for res in ["cpu", "memory", "disk", "network"])
        ]

        for metric_name in resource_metrics:
            stats = self.metrics_collector.get_metric_statistics(
                metric_name, window_seconds
            )

            if stats and stats["mean"] > 80:  # > 80% utilization
                severity = (stats["mean"] - 80) / 2  # Scale 80-100% to 0-10

                bottleneck = BottleneckAnalysis(
                    component_name=metric_name,
                    bottleneck_type="resource",
                    severity_score=severity,
                    impact_description=f"High resource utilization: {stats['mean']:.1f}%",
                    suggested_solutions=[
                        "Scale up resources",
                        "Optimize resource usage",
                        "Implement resource pooling",
                        "Add more capacity",
                    ],
                    measurement_data=stats,
                )
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _detect_queue_bottlenecks(
        self, window_seconds: int
    ) -> List[BottleneckAnalysis]:
        """Detect queue-based bottlenecks"""
        bottlenecks = []

        # Look for growing queues
        queue_metrics = [
            name
            for name in self.metrics_collector.metrics.keys()
            if "queue" in name.lower() or "backlog" in name.lower()
        ]

        for metric_name in queue_metrics:
            stats = self.metrics_collector.get_metric_statistics(
                metric_name, window_seconds
            )

            if stats and stats["mean"] > 100:  # Queue length > 100
                severity = min(stats["mean"] / 100, 10.0)

                bottleneck = BottleneckAnalysis(
                    component_name=metric_name.replace("_queue_length", ""),
                    bottleneck_type="queue",
                    severity_score=severity,
                    impact_description=f"Queue buildup detected: {stats['mean']:.0f} items",
                    suggested_solutions=[
                        "Increase processing capacity",
                        "Implement queue prioritization",
                        "Add more workers",
                        "Optimize processing speed",
                    ],
                    measurement_data=stats,
                )
                bottlenecks.append(bottleneck)

        return bottlenecks


class OptimizationEngine:
    """Generates optimization recommendations"""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        bottleneck_detector: BottleneckDetector,
    ):
        self.metrics_collector = metrics_collector
        self.bottleneck_detector = bottleneck_detector
        self.optimization_history: List[OptimizationRecommendation] = []

    def generate_recommendations(
        self, window_seconds: int = 300
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze bottlenecks
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks(window_seconds)

        # Generate recommendations for each bottleneck
        for bottleneck in bottlenecks:
            rec = self._create_recommendation_for_bottleneck(bottleneck)
            if rec:
                recommendations.append(rec)

        # Generate proactive recommendations
        proactive_recs = self._generate_proactive_recommendations(window_seconds)
        recommendations.extend(proactive_recs)

        # Sort by priority score
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _create_recommendation_for_bottleneck(
        self, bottleneck: BottleneckAnalysis
    ) -> Optional[OptimizationRecommendation]:
        """Create optimization recommendation for a specific bottleneck"""

        strategy_map = {
            "latency": OptimizationStrategy.CACHE_OPTIMIZATION,
            "throughput": OptimizationStrategy.SCALE_UP,
            "resource": OptimizationStrategy.RESOURCE_REALLOCATION,
            "queue": OptimizationStrategy.LOAD_BALANCE,
        }

        strategy = strategy_map.get(
            bottleneck.bottleneck_type, OptimizationStrategy.SCALE_UP
        )

        return OptimizationRecommendation(
            id=f"opt_{int(time.time())}_{bottleneck.component_name}",
            strategy=strategy,
            title=f"Optimize {bottleneck.component_name} {bottleneck.bottleneck_type}",
            description=f"Address {bottleneck.bottleneck_type} bottleneck in {bottleneck.component_name}. {bottleneck.impact_description}",
            expected_improvement=min(
                bottleneck.severity_score * 10, 50
            ),  # Up to 50% improvement
            implementation_effort="medium",
            risk_level="low" if bottleneck.severity_score < 5 else "medium",
            priority_score=bottleneck.severity_score,
            affected_components=[bottleneck.component_name],
            timestamp=time.time(),
        )

    def _generate_proactive_recommendations(
        self, window_seconds: int
    ) -> List[OptimizationRecommendation]:
        """Generate proactive optimization recommendations"""
        recommendations = []

        # Analyze trends for proactive optimizations
        for metric_name in self.metrics_collector.metrics.keys():
            if self._is_trending_upward(metric_name, window_seconds):
                rec = self._create_proactive_recommendation(metric_name)
                if rec:
                    recommendations.append(rec)

        return recommendations

    def _is_trending_upward(self, metric_name: str, window_seconds: int) -> bool:
        """Check if metric is trending upward"""
        # Get metrics from two time windows for comparison
        end_time = time.time()
        recent_start = end_time - window_seconds // 2
        older_start = end_time - window_seconds

        recent_metrics = self.metrics_collector.get_metrics(
            metric_name, recent_start, end_time
        )
        older_metrics = self.metrics_collector.get_metrics(
            metric_name, older_start, recent_start
        )

        if not recent_metrics or not older_metrics:
            return False

        recent_avg = sum(m.value for m in recent_metrics) / len(recent_metrics)
        older_avg = sum(m.value for m in older_metrics) / len(older_metrics)

        # Consider it trending if recent average is 20% higher
        return recent_avg > older_avg * 1.2

    def _create_proactive_recommendation(
        self, metric_name: str
    ) -> Optional[OptimizationRecommendation]:
        """Create proactive recommendation for trending metric"""

        if "latency" in metric_name.lower():
            return OptimizationRecommendation(
                id=f"proactive_{int(time.time())}_{metric_name}",
                strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                title=f"Proactive latency optimization for {metric_name}",
                description=f"Latency trending upward for {metric_name}. Consider adding caching or optimization.",
                expected_improvement=15.0,
                implementation_effort="medium",
                risk_level="low",
                priority_score=3.0,
                affected_components=[metric_name.replace("_latency", "")],
                timestamp=time.time(),
            )

        elif "error" in metric_name.lower():
            return OptimizationRecommendation(
                id=f"proactive_{int(time.time())}_{metric_name}",
                strategy=OptimizationStrategy.WORKFLOW_RESTRUCTURE,
                title=f"Error rate mitigation for {metric_name}",
                description=f"Error rate increasing for {metric_name}. Review and improve error handling.",
                expected_improvement=25.0,
                implementation_effort="high",
                risk_level="medium",
                priority_score=7.0,
                affected_components=[metric_name.replace("_error_rate", "")],
                timestamp=time.time(),
            )

        return None


class WorkflowMonitor:
    """Main workflow monitoring and optimization coordinator"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.bottleneck_detector = BottleneckDetector(self.metrics_collector)
        self.optimization_engine = OptimizationEngine(
            self.metrics_collector, self.bottleneck_detector
        )

        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._setup_default_collectors()
        self._setup_default_alerts()

    def start(self):
        """Start all monitoring components"""
        self.metrics_collector.start()
        self.alert_manager.start()
        logging.info("Workflow monitor started")

    def stop(self):
        """Stop all monitoring components"""
        self.metrics_collector.stop()
        self.alert_manager.stop()
        logging.info("Workflow monitor stopped")

    def _setup_default_collectors(self):
        """Setup default metrics collectors"""

        def system_metrics_collector() -> List[Metric]:
            """Collect basic system metrics"""
            import psutil

            metrics = []
            timestamp = time.time()

            # CPU utilization
            cpu_percent = psutil.cpu_percent()
            metrics.append(
                Metric(
                    name="system_cpu_utilization",
                    type=MetricType.RESOURCE_UTILIZATION,
                    value=cpu_percent,
                    timestamp=timestamp,
                    unit="percent",
                )
            )

            # Memory utilization
            memory = psutil.virtual_memory()
            metrics.append(
                Metric(
                    name="system_memory_utilization",
                    type=MetricType.RESOURCE_UTILIZATION,
                    value=memory.percent,
                    timestamp=timestamp,
                    unit="percent",
                )
            )

            return metrics

        self.metrics_collector.add_collector(system_metrics_collector)

    def _setup_default_alerts(self):
        """Setup default alert rules"""

        # High CPU utilization
        self.alert_manager.add_alert_rule(
            metric_name="system_cpu_utilization",
            threshold=80.0,
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            description="System CPU utilization is high",
        )

        # High memory utilization
        self.alert_manager.add_alert_rule(
            metric_name="system_memory_utilization",
            threshold=95.0,
            condition="greater_than",
            severity=AlertSeverity.CRITICAL,
            description="System memory utilization is critically high",
        )

    def add_baseline(
        self,
        metric_name: str,
        baseline_value: float,
        acceptable_deviation: float = 0.2,
        measurement_window: int = 100,
    ):
        """Add performance baseline for comparison"""

        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            acceptable_deviation=acceptable_deviation,
            measurement_window=measurement_window,
            last_updated=time.time(),
        )

        self.baselines[metric_name] = baseline

        # Add alert rule for baseline deviation
        threshold = baseline_value * (1 + acceptable_deviation)
        self.alert_manager.add_alert_rule(
            metric_name=metric_name,
            threshold=threshold,
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            description=f"Metric {metric_name} deviating from baseline",
        )

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""

        # Start with perfect score
        health_score = 100.0

        # Deduct points for active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        for alert in active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                health_score -= 25
            elif alert.severity == AlertSeverity.ERROR:
                health_score -= 15
            elif alert.severity == AlertSeverity.WARNING:
                health_score -= 5

        # Deduct points for bottlenecks
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks()
        for bottleneck in bottlenecks:
            health_score -= bottleneck.severity_score

        # Ensure score stays within bounds
        return max(0.0, min(100.0, health_score))

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""

        recommendations = self.optimization_engine.generate_recommendations()
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "timestamp": time.time(),
            "system_health_score": self.get_system_health_score(),
            "active_alerts_count": len(active_alerts),
            "critical_alerts": [
                a for a in active_alerts if a.severity == AlertSeverity.CRITICAL
            ],
            "bottlenecks_detected": len(bottlenecks),
            "top_bottlenecks": bottlenecks[:5],
            "optimization_recommendations": recommendations[:10],
            "high_priority_recommendations": [
                r for r in recommendations if r.priority_score > 7.0
            ],
            "performance_summary": self._get_performance_summary(),
        }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""

        summary = {}

        # Get key metrics statistics
        key_metrics = ["system_cpu_utilization", "system_memory_utilization"]

        for metric_name in key_metrics:
            if metric_name in self.metrics_collector.metrics:
                stats = self.metrics_collector.get_metric_statistics(metric_name)
                summary[metric_name] = stats

        return summary

    def simulate_load(self, duration_seconds: int = 60):
        """Simulate system load for testing (demonstration purposes)"""

        def load_simulator():
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                # Simulate varying metrics
                timestamp = time.time()

                # Simulate application latency
                base_latency = 100 + np.random.normal(0, 20)
                load_factor = (
                    1 + (time.time() - start_time) / duration_seconds
                )  # Increase over time
                latency = base_latency * load_factor

                self.metrics_collector.record_metric(
                    Metric(
                        name="app_request_latency",
                        type=MetricType.LATENCY,
                        value=latency,
                        timestamp=timestamp,
                        unit="milliseconds",
                    )
                )

                # Simulate throughput
                throughput = max(
                    1, 50 - (time.time() - start_time) / 2
                )  # Decrease over time
                self.metrics_collector.record_metric(
                    Metric(
                        name="app_throughput",
                        type=MetricType.THROUGHPUT,
                        value=throughput,
                        timestamp=timestamp,
                        unit="requests_per_second",
                    )
                )

                # Simulate error rate
                error_rate = min(
                    20, (time.time() - start_time) / 3
                )  # Increase over time
                self.metrics_collector.record_metric(
                    Metric(
                        name="app_error_rate",
                        type=MetricType.ERROR_RATE,
                        value=error_rate,
                        timestamp=timestamp,
                        unit="percent",
                    )
                )

                time.sleep(1)

        # Run simulation in background
        import threading

        thread = threading.Thread(target=load_simulator, daemon=True)
        thread.start()

        logging.info(f"Load simulation started for {duration_seconds} seconds")


# Example notification handlers
def console_notification_handler(alert: Alert):
    """Simple console notification handler"""
    print(f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
    print(f"  Description: {alert.description}")
    print(f"  Current Value: {alert.current_value}, Threshold: {alert.threshold}")
    print(f"  Time: {datetime.fromtimestamp(alert.timestamp)}")


def log_notification_handler(alert: Alert):
    """Log-based notification handler"""
    logging.warning(f"Alert triggered: {alert.title} - {alert.description}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create workflow monitor
    monitor = WorkflowMonitor()

    # Add notification handlers
    monitor.alert_manager.add_notification_handler(console_notification_handler)
    monitor.alert_manager.add_notification_handler(log_notification_handler)

    # Add custom alert rules
    monitor.alert_manager.add_alert_rule(
        metric_name="app_request_latency",
        threshold=500.0,  # 500ms
        condition="greater_than",
        severity=AlertSeverity.WARNING,
        description="Application latency is high",
    )

    monitor.alert_manager.add_alert_rule(
        metric_name="app_error_rate",
        threshold=10.0,  # 10%
        condition="greater_than",
        severity=AlertSeverity.ERROR,
        description="Application error rate is too high",
    )

    # Add performance baselines
    monitor.add_baseline("app_request_latency", 150.0, 0.3)  # 150ms ± 30%
    monitor.add_baseline("app_throughput", 45.0, 0.2)  # 45 rps ± 20%

    # Start monitoring
    monitor.start()

    print("Workflow monitor started. Simulating load...")

    # Simulate load to trigger alerts and generate recommendations
    monitor.simulate_load(30)

    # Wait and check results
    time.sleep(35)

    # Generate optimization report
    report = monitor.get_optimization_report()

    print(f"\n=== OPTIMIZATION REPORT ===")
    print(f"System Health Score: {report['system_health_score']:.1f}/100")
    print(f"Active Alerts: {report['active_alerts_count']}")
    print(f"Bottlenecks Detected: {report['bottlenecks_detected']}")
    print(
        f"Optimization Recommendations: {len(report['optimization_recommendations'])}"
    )

    print(f"\n=== TOP RECOMMENDATIONS ===")
    for rec in report["optimization_recommendations"][:3]:
        print(f"- {rec.title}")
        print(f"  Expected Improvement: {rec.expected_improvement:.1f}%")
        print(f"  Priority Score: {rec.priority_score:.1f}")
        print(f"  Implementation Effort: {rec.implementation_effort}")

    # Stop monitoring
    monitor.stop()
