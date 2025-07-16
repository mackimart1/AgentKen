"""
User Interface for Feedback Collection
Provides web-based and programmatic interfaces for collecting user feedback.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, session
from feedback_system import (
    FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, ContinuousLearningEngine,
    FeedbackType, TaskExecution, UserFeedback, ComponentType, TaskOutcome
)


class FeedbackWebInterface:
    """Web interface for feedback collection and management"""
    
    def __init__(self, feedback_collector: FeedbackCollector, 
                 feedback_analyzer: FeedbackAnalyzer,
                 learning_engine: ContinuousLearningEngine,
                 host: str = "localhost", port: int = 5002):
        self.collector = feedback_collector
        self.analyzer = feedback_analyzer
        self.learning_engine = learning_engine
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.secret_key = "feedback_system_secret_key"
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main feedback dashboard"""
            return render_template('feedback_dashboard.html')
        
        @self.app.route('/feedback')
        def feedback_form():
            """Feedback collection form"""
            task_id = request.args.get('task_id')
            user_id = request.args.get('user_id', 'anonymous')
            return render_template('feedback_form.html', task_id=task_id, user_id=user_id)
        
        @self.app.route('/api/pending_tasks/<user_id>')
        def api_pending_tasks(user_id):
            """Get pending tasks for a user"""
            tasks = self.collector.get_pending_tasks_for_user(user_id)
            return jsonify([{
                'id': task.id,
                'component_id': task.component_id,
                'operation': task.operation,
                'timestamp': task.timestamp,
                'outcome': task.outcome.value,
                'execution_time': task.execution_time
            } for task in tasks])
        
        @self.app.route('/api/submit_feedback', methods=['POST'])
        def api_submit_feedback():
            """Submit user feedback"""
            data = request.json
            
            try:
                feedback_type = FeedbackType(data['feedback_type'])
                
                # Prepare feedback data
                feedback_data = {
                    'rating': data.get('rating'),
                    'binary_score': data.get('binary_score'),
                    'category': data.get('category'),
                    'text_feedback': data.get('text_feedback'),
                    'detailed_scores': data.get('detailed_scores', {}),
                    'tags': data.get('tags', []),
                    'context': data.get('context', {})
                }
                
                # Remove None values
                feedback_data = {k: v for k, v in feedback_data.items() if v is not None}
                
                feedback = self.collector.collect_feedback(
                    task_execution_id=data['task_execution_id'],
                    user_id=data['user_id'],
                    feedback_type=feedback_type,
                    **feedback_data
                )
                
                return jsonify({
                    'status': 'success',
                    'feedback_id': feedback.id,
                    'message': 'Feedback submitted successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 400
        
        @self.app.route('/api/component_performance/<component_id>')
        def api_component_performance(component_id):
            """Get component performance analysis"""
            hours = request.args.get('hours', 168, type=int)
            profile = self.analyzer.analyze_component_performance(component_id, hours)
            return jsonify({
                'component_id': profile.component_id,
                'total_executions': profile.total_executions,
                'total_feedback_count': profile.total_feedback_count,
                'average_rating': profile.average_rating,
                'satisfaction_rate': profile.satisfaction_rate,
                'improvement_trend': profile.improvement_trend,
                'strengths': profile.strengths,
                'weaknesses': profile.weaknesses,
                'recommendations': profile.recommendations,
                'last_updated': profile.last_updated
            })
        
        @self.app.route('/api/learning_insights')
        def api_learning_insights():
            """Get learning insights"""
            component_id = request.args.get('component_id')
            insights = self.analyzer.generate_learning_insights(component_id)
            return jsonify([{
                'component_id': insight.component_id,
                'component_type': insight.component_type.value,
                'operation': insight.operation,
                'insight_type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'impact_score': insight.impact_score,
                'recommendations': insight.recommendations,
                'timestamp': insight.timestamp
            } for insight in insights])
        
        @self.app.route('/api/apply_learning', methods=['POST'])
        def api_apply_learning():
            """Apply learning strategies"""
            data = request.json
            component_id = data.get('component_id')
            
            adaptations = self.learning_engine.apply_learning(component_id)
            return jsonify({
                'status': 'success',
                'adaptations_applied': len(adaptations),
                'adaptations': adaptations
            })
        
        @self.app.route('/api/feedback_stats')
        def api_feedback_stats():
            """Get overall feedback statistics"""
            # Get recent feedback
            feedback_list = self.collector.storage.get_feedback(limit=1000)
            
            # Calculate stats
            total_feedback = len(feedback_list)
            rating_feedback = [f for f in feedback_list if f.rating is not None]
            binary_feedback = [f for f in feedback_list if f.binary_score is not None]
            
            avg_rating = sum(f.rating for f in rating_feedback) / len(rating_feedback) if rating_feedback else 0
            positive_binary = sum(1 for f in binary_feedback if f.binary_score) if binary_feedback else 0
            binary_satisfaction = (positive_binary / len(binary_feedback) * 100) if binary_feedback else 0
            
            # Recent feedback trend
            recent_feedback = [f for f in feedback_list if f.timestamp > time.time() - 86400]  # Last 24 hours
            
            return jsonify({
                'total_feedback': total_feedback,
                'average_rating': avg_rating,
                'binary_satisfaction_rate': binary_satisfaction,
                'recent_feedback_count': len(recent_feedback),
                'feedback_types': {
                    'rating': len(rating_feedback),
                    'binary': len(binary_feedback),
                    'text': len([f for f in feedback_list if f.text_feedback]),
                    'detailed': len([f for f in feedback_list if f.detailed_scores])
                }
            })
        
        @self.app.route('/api/user_feedback_history/<user_id>')
        def api_user_feedback_history(user_id):
            """Get feedback history for a specific user"""
            feedback_list = self.collector.storage.get_feedback(user_id=user_id)
            return jsonify([{
                'id': feedback.id,
                'task_execution_id': feedback.task_execution_id,
                'feedback_type': feedback.feedback_type.value,
                'rating': feedback.rating,
                'binary_score': feedback.binary_score,
                'category': feedback.category,
                'text_feedback': feedback.text_feedback,
                'timestamp': feedback.timestamp,
                'tags': feedback.tags
            } for feedback in feedback_list])
    
    def run(self, debug: bool = False):
        """Run the feedback web interface"""
        print(f"Starting feedback interface at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


class FeedbackPromptSystem:
    """System for prompting users for feedback at appropriate times"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.collector = feedback_collector
        self.prompt_strategies = {
            'immediate': self._immediate_prompt,
            'delayed': self._delayed_prompt,
            'batch': self._batch_prompt,
            'smart': self._smart_prompt
        }
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Set feedback preferences for a user"""
        self.user_preferences[user_id] = preferences
    
    def should_prompt_for_feedback(self, execution: TaskExecution, strategy: str = 'smart') -> bool:
        """Determine if user should be prompted for feedback"""
        if strategy not in self.prompt_strategies:
            strategy = 'smart'
        
        return self.prompt_strategies[strategy](execution)
    
    def _immediate_prompt(self, execution: TaskExecution) -> bool:
        """Always prompt immediately after task completion"""
        return True
    
    def _delayed_prompt(self, execution: TaskExecution) -> bool:
        """Prompt after a delay (e.g., for batch processing)"""
        # Check if enough time has passed since execution
        return time.time() - execution.timestamp > 300  # 5 minutes
    
    def _batch_prompt(self, execution: TaskExecution) -> bool:
        """Prompt when user has multiple completed tasks"""
        if not execution.user_id:
            return False
        
        user_tasks = self.collector.get_pending_tasks_for_user(execution.user_id)
        return len(user_tasks) >= 3  # Prompt when 3+ tasks are pending feedback
    
    def _smart_prompt(self, execution: TaskExecution) -> bool:
        """Smart prompting based on task importance and user behavior"""
        # Don't prompt for very quick tasks
        if execution.execution_time < 1.0:
            return False
        
        # Always prompt for failures
        if execution.outcome in [TaskOutcome.FAILURE, TaskOutcome.ERROR]:
            return True
        
        # Prompt for long-running tasks
        if execution.execution_time > 30.0:
            return True
        
        # Check user preferences
        if execution.user_id in self.user_preferences:
            prefs = self.user_preferences[execution.user_id]
            if prefs.get('feedback_frequency') == 'minimal':
                return False
            elif prefs.get('feedback_frequency') == 'all':
                return True
        
        # Default: prompt for 30% of tasks
        import random
        return random.random() < 0.3
    
    def generate_feedback_prompt(self, execution: TaskExecution) -> Dict[str, Any]:
        """Generate a contextual feedback prompt"""
        prompt_data = {
            'task_id': execution.id,
            'component_name': execution.component_id,
            'operation': execution.operation,
            'outcome': execution.outcome.value,
            'execution_time': execution.execution_time,
            'timestamp': execution.timestamp
        }
        
        # Customize prompt based on outcome
        if execution.outcome == TaskOutcome.SUCCESS:
            prompt_data['message'] = f"How satisfied are you with the {execution.operation} results?"
            prompt_data['suggested_feedback_type'] = 'rating'
        elif execution.outcome == TaskOutcome.FAILURE:
            prompt_data['message'] = f"We noticed the {execution.operation} didn't work as expected. Can you help us improve?"
            prompt_data['suggested_feedback_type'] = 'detailed'
        else:
            prompt_data['message'] = f"Please share your experience with {execution.operation}"
            prompt_data['suggested_feedback_type'] = 'rating'
        
        return prompt_data


def create_feedback_templates():
    """Create HTML templates for the feedback interface"""
    
    dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentKen Feedback Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .content-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .insights-list {
            list-style: none;
            padding: 0;
        }
        .insight-item {
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            background-color: #f8f9fa;
            border-radius: 0 5px 5px 0;
        }
        .insight-title {
            font-weight: bold;
            color: #333;
        }
        .insight-description {
            color: #666;
            margin-top: 5px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
        }
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        .btn-success {
            background-color: #28a745;
            color: white;
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
        <h1>🔄 AgentKen Feedback Dashboard</h1>
        <p>Continuous learning through user feedback</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="total-feedback">--</div>
            <div class="stat-label">Total Feedback</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="avg-rating">--</div>
            <div class="stat-label">Average Rating</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="satisfaction-rate">--%</div>
            <div class="stat-label">Satisfaction Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="recent-feedback">--</div>
            <div class="stat-label">Recent Feedback (24h)</div>
        </div>
    </div>

    <div class="content-section">
        <h3>📊 Feedback Distribution</h3>
        <canvas id="feedbackChart" width="400" height="200"></canvas>
    </div>

    <div class="content-section">
        <h3>🧠 Learning Insights</h3>
        <ul class="insights-list" id="insights-list">
            <li>Loading insights...</li>
        </ul>
        <button class="btn btn-success" onclick="applyLearning()">Apply Learning Strategies</button>
    </div>

    <div class="content-section">
        <h3>🎯 Component Performance</h3>
        <div id="component-performance">
            <p>Select a component to view performance details</p>
        </div>
    </div>

    <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>

    <script>
        let feedbackChart;

        function loadFeedbackStats() {
            $.get('/api/feedback_stats', function(data) {
                $('#total-feedback').text(data.total_feedback);
                $('#avg-rating').text(data.average_rating.toFixed(1));
                $('#satisfaction-rate').text(data.binary_satisfaction_rate.toFixed(1) + '%');
                $('#recent-feedback').text(data.recent_feedback_count);
                
                updateFeedbackChart(data.feedback_types);
            });
        }

        function updateFeedbackChart(feedbackTypes) {
            const ctx = document.getElementById('feedbackChart').getContext('2d');
            
            if (feedbackChart) {
                feedbackChart.destroy();
            }
            
            feedbackChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Rating', 'Binary', 'Text', 'Detailed'],
                    datasets: [{
                        data: [
                            feedbackTypes.rating,
                            feedbackTypes.binary,
                            feedbackTypes.text,
                            feedbackTypes.detailed
                        ],
                        backgroundColor: [
                            '#667eea',
                            '#764ba2',
                            '#f093fb',
                            '#f5576c'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        function loadLearningInsights() {
            $.get('/api/learning_insights', function(insights) {
                const insightsList = $('#insights-list');
                insightsList.empty();
                
                if (insights.length === 0) {
                    insightsList.append('<li>No insights available yet. More feedback needed.</li>');
                    return;
                }
                
                insights.forEach(insight => {
                    const insightHtml = `
                        <li class="insight-item">
                            <div class="insight-title">${insight.component_id} - ${insight.insight_type}</div>
                            <div class="insight-description">${insight.description}</div>
                            <small>Confidence: ${(insight.confidence * 100).toFixed(1)}% | Impact: ${insight.impact_score}/10</small>
                        </li>
                    `;
                    insightsList.append(insightHtml);
                });
            });
        }

        function applyLearning() {
            $.post('/api/apply_learning', {}, function(response) {
                alert(`Applied ${response.adaptations_applied} learning adaptations`);
                loadLearningInsights();
            });
        }

        function refreshDashboard() {
            loadFeedbackStats();
            loadLearningInsights();
        }

        // Initialize dashboard
        $(document).ready(function() {
            refreshDashboard();
            
            // Auto-refresh every 60 seconds
            setInterval(refreshDashboard, 60000);
        });
    </script>
</body>
</html>
    """
    
    feedback_form_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Provide Feedback - AgentKen</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        .form-control {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        .rating-stars {
            display: flex;
            gap: 5px;
            margin-top: 5px;
        }
        .star {
            font-size: 24px;
            color: #ddd;
            cursor: pointer;
            transition: color 0.2s;
        }
        .star.active {
            color: #ffd700;
        }
        .binary-buttons {
            display: flex;
            gap: 10px;
            margin-top: 5px;
        }
        .binary-btn {
            padding: 10px 20px;
            border: 2px solid #ddd;
            background: white;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .binary-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .feedback-type-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .type-btn {
            padding: 8px 16px;
            border: 2px solid #ddd;
            background: white;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .type-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .feedback-section {
            display: none;
        }
        .feedback-section.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>📝 Share Your Feedback</h2>
            <p>Help us improve AgentKen with your valuable feedback</p>
        </div>

        <form id="feedback-form">
            <input type="hidden" id="task-id" value="{{ task_id }}">
            <input type="hidden" id="user-id" value="{{ user_id }}">

            <div class="form-group">
                <label class="form-label">Feedback Type</label>
                <div class="feedback-type-selector">
                    <button type="button" class="type-btn active" data-type="rating">⭐ Rating</button>
                    <button type="button" class="type-btn" data-type="binary">👍 Quick</button>
                    <button type="button" class="type-btn" data-type="text">💬 Text</button>
                    <button type="button" class="type-btn" data-type="detailed">📊 Detailed</button>
                </div>
            </div>

            <!-- Rating Feedback -->
            <div class="feedback-section active" id="rating-section">
                <div class="form-group">
                    <label class="form-label">How would you rate this task outcome?</label>
                    <div class="rating-stars" id="rating-stars">
                        <span class="star" data-rating="1">⭐</span>
                        <span class="star" data-rating="2">⭐</span>
                        <span class="star" data-rating="3">⭐</span>
                        <span class="star" data-rating="4">⭐</span>
                        <span class="star" data-rating="5">⭐</span>
                    </div>
                    <input type="hidden" id="rating-value" value="">
                </div>
            </div>

            <!-- Binary Feedback -->
            <div class="feedback-section" id="binary-section">
                <div class="form-group">
                    <label class="form-label">Was this task outcome satisfactory?</label>
                    <div class="binary-buttons">
                        <button type="button" class="binary-btn" data-value="true">👍 Yes</button>
                        <button type="button" class="binary-btn" data-value="false">👎 No</button>
                    </div>
                    <input type="hidden" id="binary-value" value="">
                </div>
            </div>

            <!-- Text Feedback -->
            <div class="feedback-section" id="text-section">
                <div class="form-group">
                    <label class="form-label">Please share your thoughts</label>
                    <textarea class="form-control" id="text-feedback" rows="4" 
                              placeholder="Tell us about your experience..."></textarea>
                </div>
            </div>

            <!-- Detailed Feedback -->
            <div class="feedback-section" id="detailed-section">
                <div class="form-group">
                    <label class="form-label">Rate different aspects:</label>
                    
                    <div style="margin: 10px 0;">
                        <label>Accuracy:</label>
                        <div class="rating-stars" data-aspect="accuracy">
                            <span class="star" data-rating="1">⭐</span>
                            <span class="star" data-rating="2">⭐</span>
                            <span class="star" data-rating="3">⭐</span>
                            <span class="star" data-rating="4">⭐</span>
                            <span class="star" data-rating="5">⭐</span>
                        </div>
                    </div>
                    
                    <div style="margin: 10px 0;">
                        <label>Speed:</label>
                        <div class="rating-stars" data-aspect="speed">
                            <span class="star" data-rating="1">⭐</span>
                            <span class="star" data-rating="2">⭐</span>
                            <span class="star" data-rating="3">⭐</span>
                            <span class="star" data-rating="4">⭐</span>
                            <span class="star" data-rating="5">⭐</span>
                        </div>
                    </div>
                    
                    <div style="margin: 10px 0;">
                        <label>Usefulness:</label>
                        <div class="rating-stars" data-aspect="usefulness">
                            <span class="star" data-rating="1">⭐</span>
                            <span class="star" data-rating="2">⭐</span>
                            <span class="star" data-rating="3">⭐</span>
                            <span class="star" data-rating="4">⭐</span>
                            <span class="star" data-rating="5">⭐</span>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Additional Comments</label>
                    <textarea class="form-control" id="detailed-comments" rows="3" 
                              placeholder="Any additional feedback..."></textarea>
                </div>
            </div>

            <div class="form-group" style="text-align: center; margin-top: 30px;">
                <button type="submit" class="btn btn-primary">Submit Feedback</button>
                <button type="button" class="btn btn-secondary" onclick="window.close()">Cancel</button>
            </div>
        </form>
    </div>

    <script>
        let currentFeedbackType = 'rating';
        let detailedScores = {};

        // Feedback type selection
        $('.type-btn').click(function() {
            $('.type-btn').removeClass('active');
            $(this).addClass('active');
            
            currentFeedbackType = $(this).data('type');
            
            $('.feedback-section').removeClass('active');
            $('#' + currentFeedbackType + '-section').addClass('active');
        });

        // Rating stars
        $('.rating-stars .star').click(function() {
            const rating = $(this).data('rating');
            const container = $(this).parent();
            const aspect = container.data('aspect');
            
            // Update visual state
            container.find('.star').removeClass('active');
            for (let i = 1; i <= rating; i++) {
                container.find(`[data-rating="${i}"]`).addClass('active');
            }
            
            if (aspect) {
                // Detailed rating
                detailedScores[aspect] = rating;
            } else {
                // Simple rating
                $('#rating-value').val(rating);
            }
        });

        // Binary buttons
        $('.binary-btn').click(function() {
            $('.binary-btn').removeClass('active');
            $(this).addClass('active');
            $('#binary-value').val($(this).data('value'));
        });

        // Form submission
        $('#feedback-form').submit(function(e) {
            e.preventDefault();
            
            const feedbackData = {
                task_execution_id: $('#task-id').val(),
                user_id: $('#user-id').val(),
                feedback_type: currentFeedbackType
            };

            // Add type-specific data
            if (currentFeedbackType === 'rating') {
                feedbackData.rating = parseFloat($('#rating-value').val());
            } else if (currentFeedbackType === 'binary') {
                feedbackData.binary_score = $('#binary-value').val() === 'true';
            } else if (currentFeedbackType === 'text') {
                feedbackData.text_feedback = $('#text-feedback').val();
            } else if (currentFeedbackType === 'detailed') {
                feedbackData.detailed_scores = detailedScores;
                feedbackData.text_feedback = $('#detailed-comments').val();
            }

            // Submit feedback
            $.ajax({
                url: '/api/submit_feedback',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(feedbackData),
                success: function(response) {
                    alert('Thank you for your feedback!');
                    window.close();
                },
                error: function(xhr) {
                    alert('Error submitting feedback: ' + xhr.responseJSON.message);
                }
            });
        });
    </script>
</body>
</html>
    """
    
    return dashboard_template, feedback_form_template


if __name__ == "__main__":
    # Example usage
    from feedback_system import FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, ContinuousLearningEngine
    
    # Initialize components
    storage = FeedbackStorage("feedback_interface_test.db")
    collector = FeedbackCollector(storage)
    analyzer = FeedbackAnalyzer(storage)
    learning_engine = ContinuousLearningEngine(storage, analyzer)
    
    # Create templates
    dashboard_template, feedback_form_template = create_feedback_templates()
    
    # Save templates
    import os
    templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    with open(os.path.join(templates_dir, 'feedback_dashboard.html'), 'w') as f:
        f.write(dashboard_template)
    
    with open(os.path.join(templates_dir, 'feedback_form.html'), 'w') as f:
        f.write(feedback_form_template)
    
    print("Templates created successfully!")
    
    # Start web interface
    web_interface = FeedbackWebInterface(collector, analyzer, learning_engine)
    web_interface.run(debug=True)