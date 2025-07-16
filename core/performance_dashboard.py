"""
Web-based Performance Dashboard for AgentKen
Provides real-time visualization of performance metrics and alerts.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
from performance_monitor import PerformanceMonitor, MetricType, ComponentType, AlertLevel


class PerformanceDashboardServer:
    """Web server for performance dashboard"""
    
    def __init__(self, monitor: PerformanceMonitor, host: str = "localhost", port: int = 5001):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('performance_dashboard.html')
        
        @self.app.route('/api/overview')
        def api_overview():
            """API endpoint for system overview"""
            time_window = request.args.get('hours', 24, type=int)
            overview = self.monitor.dashboard.generate_system_overview(time_window)
            overview['health_score'] = self.monitor.get_system_health_score()
            return jsonify(overview)
        
        @self.app.route('/api/component/<component_id>')
        def api_component(component_id):
            """API endpoint for component details"""
            time_window = request.args.get('hours', 24, type=int)
            report = self.monitor.dashboard.generate_component_report(component_id, time_window)
            return jsonify(report)
        
        @self.app.route('/api/metrics/chart')
        def api_metrics_chart():
            """API endpoint for metrics chart data"""
            component_id = request.args.get('component_id')
            metric_type = request.args.get('metric_type')
            hours = request.args.get('hours', 24, type=int)
            
            chart_data = self._generate_metrics_chart(component_id, metric_type, hours)
            return jsonify(chart_data)
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """API endpoint for active alerts"""
            alerts = self.monitor.storage.get_active_alerts()
            return jsonify([{
                'id': alert.id,
                'component_id': alert.component_id,
                'metric_type': alert.metric_type.value,
                'level': alert.level.value,
                'title': alert.title,
                'description': alert.description,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp,
                'acknowledged': alert.acknowledged
            } for alert in alerts])
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def api_acknowledge_alert(alert_id):
            """API endpoint to acknowledge an alert"""
            self.monitor.alert_manager.acknowledge_alert(alert_id)
            return jsonify({'status': 'acknowledged'})
        
        @self.app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
        def api_resolve_alert(alert_id):
            """API endpoint to resolve an alert"""
            self.monitor.alert_manager.resolve_alert(alert_id)
            return jsonify({'status': 'resolved'})
        
        @self.app.route('/api/health')
        def api_health():
            """API endpoint for system health"""
            return jsonify({
                'health_score': self.monitor.get_system_health_score(),
                'timestamp': time.time()
            })
    
    def _generate_metrics_chart(self, component_id: str, metric_type: str, hours: int) -> Dict[str, Any]:
        """Generate chart data for metrics"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        try:
            metric_type_enum = MetricType(metric_type)
        except ValueError:
            return {'error': 'Invalid metric type'}
        
        metrics = self.monitor.storage.get_metrics(
            component_id=component_id,
            metric_type=metric_type_enum,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return {'timestamps': [], 'values': [], 'unit': ''}
        
        # Sort by timestamp
        metrics.sort(key=lambda m: m.timestamp)
        
        timestamps = [datetime.fromtimestamp(m.timestamp).isoformat() for m in metrics]
        values = [m.value for m in metrics]
        unit = metrics[0].unit if metrics else ''
        
        return {
            'timestamps': timestamps,
            'values': values,
            'unit': unit,
            'component_id': component_id,
            'metric_type': metric_type
        }
    
    def run(self, debug: bool = False):
        """Run the dashboard server"""
        print(f"Starting performance dashboard at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_dashboard_template():
    """Create the HTML template for the dashboard"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentKen Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .health-score {
            font-size: 3em;
            font-weight: bold;
        }
        .health-excellent { color: #4CAF50; }
        .health-good { color: #8BC34A; }
        .health-warning { color: #FF9800; }
        .health-critical { color: #F44336; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }
        .alert-critical { border-left-color: #F44336; background-color: #ffebee; }
        .alert-error { border-left-color: #FF5722; background-color: #fff3e0; }
        .alert-warning { border-left-color: #FF9800; background-color: #fff8e1; }
        .alert-info { border-left-color: #2196F3; background-color: #e3f2fd; }
        .alert-actions {
            margin-top: 10px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        .btn-primary { background-color: #2196F3; color: white; }
        .btn-success { background-color: #4CAF50; color: white; }
        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .component-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .component-stats {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.8em;
            color: #666;
        }
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 20px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AgentKen Performance Dashboard</h1>
        <p>Real-time monitoring of agents and tools performance</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value health-score" id="health-score">--</div>
            <div class="metric-label">System Health Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="total-executions">--</div>
            <div class="metric-label">Total Executions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="success-rate">--%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="avg-latency">-- ms</div>
            <div class="metric-label">Average Latency</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-alerts">--</div>
            <div class="metric-label">Active Alerts</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="total-components">--</div>
            <div class="metric-label">Total Components</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>System Latency Trend</h3>
        <div id="latency-chart" style="height: 400px;"></div>
    </div>

    <div class="chart-container">
        <h3>Success Rate Trend</h3>
        <div id="success-chart" style="height: 400px;"></div>
    </div>

    <div class="alerts-container">
        <h3>🚨 Active Alerts</h3>
        <div id="alerts-list">
            <p>Loading alerts...</p>
        </div>
    </div>

    <div class="chart-container">
        <h3>📊 Component Performance</h3>
        <div id="components-list" class="components-grid">
            <p>Loading components...</p>
        </div>
    </div>

    <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>

    <script>
        let refreshInterval;

        function updateHealthScore(score) {
            const element = document.getElementById('health-score');
            element.textContent = Math.round(score);
            
            // Update color based on score
            element.className = 'metric-value health-score ';
            if (score >= 90) element.className += 'health-excellent';
            else if (score >= 75) element.className += 'health-good';
            else if (score >= 50) element.className += 'health-warning';
            else element.className += 'health-critical';
        }

        function loadOverview() {
            $.get('/api/overview', function(data) {
                const metrics = data.system_metrics;
                
                updateHealthScore(data.health_score);
                document.getElementById('total-executions').textContent = metrics.total_executions;
                document.getElementById('success-rate').textContent = metrics.system_success_rate.toFixed(1) + '%';
                document.getElementById('avg-latency').textContent = metrics.avg_system_latency.toFixed(0) + ' ms';
                document.getElementById('active-alerts').textContent = metrics.active_alerts;
                document.getElementById('total-components').textContent = metrics.total_components;
                
                // Update components
                updateComponentsList(data.component_stats);
            });
        }

        function loadAlerts() {
            $.get('/api/alerts', function(alerts) {
                const alertsList = document.getElementById('alerts-list');
                
                if (alerts.length === 0) {
                    alertsList.innerHTML = '<p style="color: #4CAF50;">✅ No active alerts</p>';
                    return;
                }
                
                let html = '';
                alerts.forEach(alert => {
                    html += `
                        <div class="alert-item alert-${alert.level}">
                            <strong>${alert.title}</strong>
                            <p>${alert.description}</p>
                            <small>
                                Component: ${alert.component_id} | 
                                Current: ${alert.current_value.toFixed(2)} | 
                                Threshold: ${alert.threshold.toFixed(2)} |
                                Time: ${new Date(alert.timestamp * 1000).toLocaleString()}
                            </small>
                            <div class="alert-actions">
                                ${!alert.acknowledged ? `<button class="btn btn-primary" onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>` : ''}
                                <button class="btn btn-success" onclick="resolveAlert('${alert.id}')">Resolve</button>
                            </div>
                        </div>
                    `;
                });
                
                alertsList.innerHTML = html;
            });
        }

        function updateComponentsList(components) {
            const container = document.getElementById('components-list');
            
            if (Object.keys(components).length === 0) {
                container.innerHTML = '<p>No components found</p>';
                return;
            }
            
            let html = '';
            Object.entries(components).forEach(([componentId, stats]) => {
                const successRate = stats.success_rate || 0;
                const avgLatency = stats.avg_latency || 0;
                
                html += `
                    <div class="component-card">
                        <h4>${componentId}</h4>
                        <div class="component-stats">
                            <div class="stat">
                                <div class="stat-value">${stats.total_executions}</div>
                                <div class="stat-label">Executions</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">${successRate.toFixed(1)}%</div>
                                <div class="stat-label">Success Rate</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">${avgLatency.toFixed(0)}ms</div>
                                <div class="stat-label">Avg Latency</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }

        function loadLatencyChart() {
            // For demo purposes, we'll create a sample chart
            // In a real implementation, this would fetch actual metrics data
            const trace = {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'System Latency',
                line: { color: '#667eea' }
            };
            
            const layout = {
                title: 'System Latency Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Latency (ms)' },
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            Plotly.newPlot('latency-chart', [trace], layout);
        }

        function loadSuccessChart() {
            // For demo purposes, we'll create a sample chart
            const trace = {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Success Rate',
                line: { color: '#4CAF50' }
            };
            
            const layout = {
                title: 'Success Rate Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Success Rate (%)' },
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            Plotly.newPlot('success-chart', [trace], layout);
        }

        function acknowledgeAlert(alertId) {
            $.post(`/api/alerts/${alertId}/acknowledge`, function() {
                loadAlerts();
            });
        }

        function resolveAlert(alertId) {
            $.post(`/api/alerts/${alertId}/resolve`, function() {
                loadAlerts();
                loadOverview(); // Refresh overview to update alert count
            });
        }

        function refreshDashboard() {
            loadOverview();
            loadAlerts();
            loadLatencyChart();
            loadSuccessChart();
        }

        // Initialize dashboard
        $(document).ready(function() {
            refreshDashboard();
            
            // Auto-refresh every 30 seconds
            refreshInterval = setInterval(refreshDashboard, 30000);
        });

        // Cleanup on page unload
        $(window).on('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import os
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Write the template file
    template_path = os.path.join(templates_dir, 'performance_dashboard.html')
    with open(template_path, 'w') as f:
        f.write(create_dashboard_template())
    
    print(f"Dashboard template created at: {template_path}")
    
    # Example of running the dashboard
    from performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    dashboard_server = PerformanceDashboardServer(monitor)
    dashboard_server.run(debug=True)