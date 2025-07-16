"""
User Feedback Loop System for AgentKen
Collects user feedback on task outcomes and implements continuous learning to improve agent and tool behaviors.
"""

import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from collections import defaultdict, deque
import threading


class FeedbackType(Enum):
    """Types of feedback that can be provided"""
    RATING = "rating"
    BINARY = "binary"  # thumbs up/down
    CATEGORICAL = "categorical"  # predefined categories
    TEXT = "text"  # free text feedback
    DETAILED = "detailed"  # structured detailed feedback


class TaskOutcome(Enum):
    """Task outcome status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"


class ComponentType(Enum):
    """Types of system components"""
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    SYSTEM = "system"


@dataclass
class TaskExecution:
    """Represents a task execution that can receive feedback"""
    id: str
    component_id: str
    component_type: ComponentType
    operation: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    outcome: TaskOutcome
    execution_time: float
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback on a task execution"""
    id: str
    task_execution_id: str
    user_id: str
    feedback_type: FeedbackType
    rating: Optional[float] = None  # 1-5 scale for rating feedback
    binary_score: Optional[bool] = None  # True/False for binary feedback
    category: Optional[str] = None  # Category for categorical feedback
    text_feedback: Optional[str] = None  # Free text feedback
    detailed_scores: Dict[str, float] = field(default_factory=dict)  # Detailed aspect ratings
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LearningInsight:
    """Insights derived from feedback analysis"""
    component_id: str
    component_type: ComponentType
    operation: str
    insight_type: str
    description: str
    confidence: float
    impact_score: float
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentPerformanceProfile:
    """Performance profile for a component based on feedback"""
    component_id: str
    component_type: ComponentType
    total_executions: int
    total_feedback_count: int
    average_rating: float
    satisfaction_rate: float  # % of positive feedback
    improvement_trend: float  # positive = improving, negative = declining
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    last_updated: float = field(default_factory=time.time)


class FeedbackStorage:
    """Persistent storage for feedback data"""
    
    def __init__(self, db_path: str = "feedback_system.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Task executions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_executions (
                    id TEXT PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    context TEXT,
                    metadata TEXT
                )
            """)
            
            # User feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    task_execution_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating REAL,
                    binary_score INTEGER,
                    category TEXT,
                    text_feedback TEXT,
                    detailed_scores TEXT,
                    timestamp REAL NOT NULL,
                    context TEXT,
                    tags TEXT,
                    FOREIGN KEY (task_execution_id) REFERENCES task_executions (id)
                )
            """)
            
            # Learning insights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id TEXT PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    impact_score REAL NOT NULL,
                    recommendations TEXT NOT NULL,
                    supporting_data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Performance profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_profiles (
                    component_id TEXT PRIMARY KEY,
                    component_type TEXT NOT NULL,
                    total_executions INTEGER NOT NULL,
                    total_feedback_count INTEGER NOT NULL,
                    average_rating REAL NOT NULL,
                    satisfaction_rate REAL NOT NULL,
                    improvement_trend REAL NOT NULL,
                    strengths TEXT NOT NULL,
                    weaknesses TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_executions_component ON task_executions(component_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_executions_timestamp ON task_executions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_task ON user_feedback(task_execution_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_component ON learning_insights(component_id)")
    
    def store_task_execution(self, execution: TaskExecution):
        """Store a task execution"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO task_executions 
                (id, component_id, component_type, operation, input_data, output_data, 
                 outcome, execution_time, timestamp, user_id, session_id, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.id,
                execution.component_id,
                execution.component_type.value,
                execution.operation,
                json.dumps(execution.input_data),
                json.dumps(execution.output_data),
                execution.outcome.value,
                execution.execution_time,
                execution.timestamp,
                execution.user_id,
                execution.session_id,
                json.dumps(execution.context),
                json.dumps(execution.metadata)
            ))
    
    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_feedback 
                (id, task_execution_id, user_id, feedback_type, rating, binary_score, 
                 category, text_feedback, detailed_scores, timestamp, context, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.id,
                feedback.task_execution_id,
                feedback.user_id,
                feedback.feedback_type.value,
                feedback.rating,
                int(feedback.binary_score) if feedback.binary_score is not None else None,
                feedback.category,
                feedback.text_feedback,
                json.dumps(feedback.detailed_scores),
                feedback.timestamp,
                json.dumps(feedback.context),
                json.dumps(feedback.tags)
            ))
    
    def get_task_executions(self, component_id: str = None, user_id: str = None, 
                           start_time: float = None, end_time: float = None, 
                           limit: int = 1000) -> List[TaskExecution]:
        """Retrieve task executions with optional filtering"""
        query = "SELECT * FROM task_executions WHERE 1=1"
        params = []
        
        if component_id:
            query += " AND component_id = ?"
            params.append(component_id)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
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
            executions = []
            
            for row in cursor.fetchall():
                execution = TaskExecution(
                    id=row[0],
                    component_id=row[1],
                    component_type=ComponentType(row[2]),
                    operation=row[3],
                    input_data=json.loads(row[4]),
                    output_data=json.loads(row[5]),
                    outcome=TaskOutcome(row[6]),
                    execution_time=row[7],
                    timestamp=row[8],
                    user_id=row[9],
                    session_id=row[10],
                    context=json.loads(row[11]) if row[11] else {},
                    metadata=json.loads(row[12]) if row[12] else {}
                )
                executions.append(execution)
            
            return executions
    
    def get_feedback(self, task_execution_id: str = None, user_id: str = None,
                    component_id: str = None, limit: int = 1000) -> List[UserFeedback]:
        """Retrieve user feedback with optional filtering"""
        if component_id:
            # Join with task_executions to filter by component
            query = """
                SELECT f.* FROM user_feedback f
                JOIN task_executions t ON f.task_execution_id = t.id
                WHERE t.component_id = ?
                ORDER BY f.timestamp DESC LIMIT ?
            """
            params = [component_id, limit]
        else:
            query = "SELECT * FROM user_feedback WHERE 1=1"
            params = []
            
            if task_execution_id:
                query += " AND task_execution_id = ?"
                params.append(task_execution_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            feedback_list = []
            
            for row in cursor.fetchall():
                feedback = UserFeedback(
                    id=row[0],
                    task_execution_id=row[1],
                    user_id=row[2],
                    feedback_type=FeedbackType(row[3]),
                    rating=row[4],
                    binary_score=bool(row[5]) if row[5] is not None else None,
                    category=row[6],
                    text_feedback=row[7],
                    detailed_scores=json.loads(row[8]) if row[8] else {},
                    timestamp=row[9],
                    context=json.loads(row[10]) if row[10] else {},
                    tags=json.loads(row[11]) if row[11] else []
                )
                feedback_list.append(feedback)
            
            return feedback_list


class FeedbackCollector:
    """Collects and manages user feedback"""
    
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.pending_tasks: Dict[str, TaskExecution] = {}
        self.feedback_handlers: List[Callable[[UserFeedback], None]] = []
        self._lock = threading.Lock()
    
    def register_task_execution(self, execution: TaskExecution):
        """Register a task execution for potential feedback"""
        with self._lock:
            self.storage.store_task_execution(execution)
            self.pending_tasks[execution.id] = execution
            
            # Clean up old pending tasks (older than 24 hours)
            cutoff_time = time.time() - 86400
            expired_tasks = [
                task_id for task_id, task in self.pending_tasks.items()
                if task.timestamp < cutoff_time
            ]
            for task_id in expired_tasks:
                del self.pending_tasks[task_id]
    
    def collect_feedback(self, task_execution_id: str, user_id: str, 
                        feedback_type: FeedbackType, **kwargs) -> UserFeedback:
        """Collect user feedback for a task execution"""
        
        # Validate task execution exists
        if task_execution_id not in self.pending_tasks:
            # Try to load from storage
            executions = self.storage.get_task_executions()
            task_exists = any(e.id == task_execution_id for e in executions)
            if not task_exists:
                raise ValueError(f"Task execution {task_execution_id} not found")
        
        # Create feedback object
        feedback = UserFeedback(
            id=str(uuid.uuid4()),
            task_execution_id=task_execution_id,
            user_id=user_id,
            feedback_type=feedback_type,
            **kwargs
        )
        
        # Store feedback
        self.storage.store_feedback(feedback)
        
        # Notify handlers
        for handler in self.feedback_handlers:
            try:
                handler(feedback)
            except Exception as e:
                logging.error(f"Feedback handler failed: {e}")
        
        logging.info(f"Collected {feedback_type.value} feedback from user {user_id} for task {task_execution_id}")
        return feedback
    
    def add_feedback_handler(self, handler: Callable[[UserFeedback], None]):
        """Add a feedback handler"""
        self.feedback_handlers.append(handler)
    
    def get_pending_tasks_for_user(self, user_id: str, limit: int = 10) -> List[TaskExecution]:
        """Get recent tasks that could benefit from feedback"""
        with self._lock:
            user_tasks = [
                task for task in self.pending_tasks.values()
                if task.user_id == user_id
            ]
            
            # Sort by timestamp (most recent first) and limit
            user_tasks.sort(key=lambda t: t.timestamp, reverse=True)
            return user_tasks[:limit]


class FeedbackAnalyzer:
    """Analyzes feedback to generate insights and learning opportunities"""
    
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.insights_cache: Dict[str, List[LearningInsight]] = {}
        self.cache_expiry = 3600  # 1 hour cache
        self.last_cache_update = 0
    
    def analyze_component_performance(self, component_id: str, 
                                    time_window_hours: int = 168) -> ComponentPerformanceProfile:
        """Analyze performance of a specific component based on feedback"""
        
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        # Get executions and feedback for the component
        executions = self.storage.get_task_executions(
            component_id=component_id,
            start_time=start_time,
            end_time=end_time
        )
        
        feedback_list = self.storage.get_feedback(component_id=component_id)
        
        if not executions:
            return ComponentPerformanceProfile(
                component_id=component_id,
                component_type=ComponentType.SYSTEM,
                total_executions=0,
                total_feedback_count=0,
                average_rating=0.0,
                satisfaction_rate=0.0,
                improvement_trend=0.0,
                strengths=[],
                weaknesses=[],
                recommendations=[]
            )
        
        # Calculate metrics
        total_executions = len(executions)
        total_feedback = len(feedback_list)
        
        # Calculate average rating
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        average_rating = np.mean(ratings) if ratings else 0.0
        
        # Calculate satisfaction rate (ratings >= 4 or positive binary feedback)
        positive_feedback = 0
        for feedback in feedback_list:
            if feedback.rating is not None and feedback.rating >= 4.0:
                positive_feedback += 1
            elif feedback.binary_score is True:
                positive_feedback += 1
        
        satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0.0
        
        # Calculate improvement trend (compare recent vs older feedback)
        improvement_trend = self._calculate_improvement_trend(feedback_list)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(feedback_list, executions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            component_id, average_rating, satisfaction_rate, strengths, weaknesses
        )
        
        return ComponentPerformanceProfile(
            component_id=component_id,
            component_type=executions[0].component_type,
            total_executions=total_executions,
            total_feedback_count=total_feedback,
            average_rating=average_rating,
            satisfaction_rate=satisfaction_rate,
            improvement_trend=improvement_trend,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def generate_learning_insights(self, component_id: str = None) -> List[LearningInsight]:
        """Generate learning insights from feedback analysis"""
        
        # Check cache
        cache_key = component_id or "all"
        if (cache_key in self.insights_cache and 
            time.time() - self.last_cache_update < self.cache_expiry):
            return self.insights_cache[cache_key]
        
        insights = []
        
        # Get feedback data
        feedback_list = self.storage.get_feedback(component_id=component_id)
        executions = self.storage.get_task_executions(component_id=component_id)
        
        if not feedback_list or not executions:
            return insights
        
        # Group by component and operation
        component_feedback = defaultdict(lambda: defaultdict(list))
        component_executions = defaultdict(lambda: defaultdict(list))
        
        for feedback in feedback_list:
            # Find corresponding execution
            execution = next((e for e in executions if e.id == feedback.task_execution_id), None)
            if execution:
                component_feedback[execution.component_id][execution.operation].append(feedback)
        
        for execution in executions:
            component_executions[execution.component_id][execution.operation].append(execution)
        
        # Generate insights for each component/operation combination
        for comp_id, operations in component_feedback.items():
            for operation, op_feedback in operations.items():
                op_executions = component_executions[comp_id][operation]
                
                # Performance insights
                performance_insight = self._analyze_performance_patterns(
                    comp_id, operation, op_feedback, op_executions
                )
                if performance_insight:
                    insights.append(performance_insight)
                
                # User satisfaction insights
                satisfaction_insight = self._analyze_satisfaction_patterns(
                    comp_id, operation, op_feedback
                )
                if satisfaction_insight:
                    insights.append(satisfaction_insight)
                
                # Error pattern insights
                error_insight = self._analyze_error_patterns(
                    comp_id, operation, op_feedback, op_executions
                )
                if error_insight:
                    insights.append(error_insight)
        
        # Cache results
        self.insights_cache[cache_key] = insights
        self.last_cache_update = time.time()
        
        return insights
    
    def _calculate_improvement_trend(self, feedback_list: List[UserFeedback]) -> float:
        """Calculate improvement trend from feedback over time"""
        if len(feedback_list) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_list, key=lambda f: f.timestamp)
        
        # Split into two halves (older vs newer)
        mid_point = len(sorted_feedback) // 2
        older_feedback = sorted_feedback[:mid_point]
        newer_feedback = sorted_feedback[mid_point:]
        
        # Calculate average ratings for each half
        older_ratings = [f.rating for f in older_feedback if f.rating is not None]
        newer_ratings = [f.rating for f in newer_feedback if f.rating is not None]
        
        if not older_ratings or not newer_ratings:
            return 0.0
        
        older_avg = np.mean(older_ratings)
        newer_avg = np.mean(newer_ratings)
        
        # Return percentage change
        return ((newer_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0.0
    
    def _identify_strengths_weaknesses(self, feedback_list: List[UserFeedback], 
                                     executions: List[TaskExecution]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses from feedback"""
        strengths = []
        weaknesses = []
        
        # Analyze text feedback for common themes
        positive_keywords = ["good", "excellent", "fast", "accurate", "helpful", "great"]
        negative_keywords = ["slow", "wrong", "bad", "error", "confusing", "poor"]
        
        positive_mentions = defaultdict(int)
        negative_mentions = defaultdict(int)
        
        for feedback in feedback_list:
            if feedback.text_feedback:
                text = feedback.text_feedback.lower()
                for keyword in positive_keywords:
                    if keyword in text:
                        positive_mentions[keyword] += 1
                for keyword in negative_keywords:
                    if keyword in text:
                        negative_mentions[keyword] += 1
        
        # Convert to strengths/weaknesses
        for keyword, count in positive_mentions.items():
            if count >= 2:  # Mentioned at least twice
                strengths.append(f"Users appreciate {keyword} performance")
        
        for keyword, count in negative_mentions.items():
            if count >= 2:
                weaknesses.append(f"Users report {keyword} issues")
        
        # Analyze ratings
        high_ratings = [f for f in feedback_list if f.rating and f.rating >= 4.5]
        low_ratings = [f for f in feedback_list if f.rating and f.rating <= 2.5]
        
        if len(high_ratings) > len(feedback_list) * 0.7:
            strengths.append("Consistently high user satisfaction")
        
        if len(low_ratings) > len(feedback_list) * 0.3:
            weaknesses.append("Significant user dissatisfaction")
        
        return strengths, weaknesses
    
    def _generate_recommendations(self, component_id: str, average_rating: float,
                                satisfaction_rate: float, strengths: List[str], 
                                weaknesses: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if average_rating < 3.0:
            recommendations.append("Critical: Review and redesign core functionality")
        elif average_rating < 4.0:
            recommendations.append("Investigate and address user pain points")
        
        if satisfaction_rate < 70:
            recommendations.append("Focus on improving user experience")
        
        if any("slow" in w.lower() for w in weaknesses):
            recommendations.append("Optimize performance and response times")
        
        if any("error" in w.lower() for w in weaknesses):
            recommendations.append("Improve error handling and reliability")
        
        if any("confusing" in w.lower() for w in weaknesses):
            recommendations.append("Enhance user interface and documentation")
        
        if not recommendations:
            recommendations.append("Continue monitoring and maintain current quality")
        
        return recommendations
    
    def _analyze_performance_patterns(self, component_id: str, operation: str,
                                    feedback_list: List[UserFeedback],
                                    executions: List[TaskExecution]) -> Optional[LearningInsight]:
        """Analyze performance patterns from feedback and execution data"""
        
        # Correlate execution time with user satisfaction
        execution_times = []
        satisfaction_scores = []
        
        for feedback in feedback_list:
            execution = next((e for e in executions if e.id == feedback.task_execution_id), None)
            if execution and feedback.rating:
                execution_times.append(execution.execution_time)
                satisfaction_scores.append(feedback.rating)
        
        if len(execution_times) < 3:
            return None
        
        # Calculate correlation
        correlation = np.corrcoef(execution_times, satisfaction_scores)[0, 1]
        
        if abs(correlation) > 0.5:  # Strong correlation
            if correlation < 0:  # Negative correlation (longer time = lower satisfaction)
                return LearningInsight(
                    component_id=component_id,
                    component_type=ComponentType.AGENT,
                    operation=operation,
                    insight_type="performance_correlation",
                    description=f"Strong negative correlation between execution time and user satisfaction (r={correlation:.2f})",
                    confidence=abs(correlation),
                    impact_score=8.0,
                    recommendations=[
                        "Optimize execution speed",
                        "Set performance targets based on user expectations",
                        "Consider async processing for long operations"
                    ],
                    supporting_data={
                        "correlation": correlation,
                        "avg_execution_time": np.mean(execution_times),
                        "avg_satisfaction": np.mean(satisfaction_scores)
                    }
                )
        
        return None
    
    def _analyze_satisfaction_patterns(self, component_id: str, operation: str,
                                     feedback_list: List[UserFeedback]) -> Optional[LearningInsight]:
        """Analyze user satisfaction patterns"""
        
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        if len(ratings) < 5:
            return None
        
        avg_rating = np.mean(ratings)
        rating_std = np.std(ratings)
        
        if rating_std > 1.5:  # High variability in ratings
            return LearningInsight(
                component_id=component_id,
                component_type=ComponentType.AGENT,
                operation=operation,
                insight_type="satisfaction_variability",
                description=f"High variability in user satisfaction (std={rating_std:.2f})",
                confidence=0.8,
                impact_score=6.0,
                recommendations=[
                    "Investigate factors causing inconsistent user experience",
                    "Standardize output quality",
                    "Implement quality assurance checks"
                ],
                supporting_data={
                    "avg_rating": avg_rating,
                    "rating_std": rating_std,
                    "sample_size": len(ratings)
                }
            )
        
        return None
    
    def _analyze_error_patterns(self, component_id: str, operation: str,
                              feedback_list: List[UserFeedback],
                              executions: List[TaskExecution]) -> Optional[LearningInsight]:
        """Analyze error patterns from feedback and executions"""
        
        error_executions = [e for e in executions if e.outcome in [TaskOutcome.FAILURE, TaskOutcome.ERROR]]
        error_feedback = []
        
        for feedback in feedback_list:
            execution = next((e for e in executions if e.id == feedback.task_execution_id), None)
            if execution and execution.outcome in [TaskOutcome.FAILURE, TaskOutcome.ERROR]:
                error_feedback.append(feedback)
        
        if len(error_executions) > len(executions) * 0.1:  # More than 10% errors
            avg_error_rating = np.mean([f.rating for f in error_feedback if f.rating]) if error_feedback else 0
            
            return LearningInsight(
                component_id=component_id,
                component_type=ComponentType.AGENT,
                operation=operation,
                insight_type="error_pattern",
                description=f"High error rate detected ({len(error_executions)}/{len(executions)} executions)",
                confidence=0.9,
                impact_score=9.0,
                recommendations=[
                    "Investigate root causes of failures",
                    "Improve error handling and recovery",
                    "Add input validation and preprocessing",
                    "Implement fallback mechanisms"
                ],
                supporting_data={
                    "error_rate": len(error_executions) / len(executions),
                    "total_errors": len(error_executions),
                    "avg_error_rating": avg_error_rating
                }
            )
        
        return None


class ContinuousLearningEngine:
    """Implements continuous learning based on user feedback"""
    
    def __init__(self, storage: FeedbackStorage, analyzer: FeedbackAnalyzer):
        self.storage = storage
        self.analyzer = analyzer
        self.learning_strategies: Dict[str, Callable] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def register_learning_strategy(self, strategy_name: str, strategy_func: Callable):
        """Register a learning strategy"""
        self.learning_strategies[strategy_name] = strategy_func
    
    def apply_learning(self, component_id: str = None) -> List[Dict[str, Any]]:
        """Apply learning strategies based on feedback insights"""
        
        insights = self.analyzer.generate_learning_insights(component_id)
        adaptations = []
        
        for insight in insights:
            # Apply relevant learning strategies
            for strategy_name, strategy_func in self.learning_strategies.items():
                try:
                    adaptation = strategy_func(insight)
                    if adaptation:
                        adaptations.append({
                            "strategy": strategy_name,
                            "insight_id": insight.component_id + "_" + insight.insight_type,
                            "adaptation": adaptation,
                            "timestamp": time.time()
                        })
                        
                        with self._lock:
                            self.adaptation_history.append(adaptations[-1])
                        
                        logging.info(f"Applied {strategy_name} learning strategy for {insight.component_id}")
                
                except Exception as e:
                    logging.error(f"Learning strategy {strategy_name} failed: {e}")
        
        return adaptations
    
    def get_adaptation_history(self, component_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history of adaptations"""
        with self._lock:
            if component_id:
                filtered_history = [
                    adaptation for adaptation in self.adaptation_history
                    if component_id in adaptation.get("insight_id", "")
                ]
                return filtered_history[-limit:]
            else:
                return self.adaptation_history[-limit:]


# Example learning strategies
def parameter_tuning_strategy(insight: LearningInsight) -> Optional[Dict[str, Any]]:
    """Learning strategy for parameter tuning based on performance feedback"""
    
    if insight.insight_type == "performance_correlation":
        correlation = insight.supporting_data.get("correlation", 0)
        avg_time = insight.supporting_data.get("avg_execution_time", 0)
        
        if correlation < -0.7 and avg_time > 5.0:  # Strong negative correlation with slow execution
            return {
                "type": "parameter_adjustment",
                "component_id": insight.component_id,
                "operation": insight.operation,
                "adjustments": {
                    "timeout_reduction": 0.8,  # Reduce timeout by 20%
                    "batch_size_increase": 1.2,  # Increase batch size by 20%
                    "cache_enabled": True
                },
                "reason": "Optimize for speed based on user feedback correlation"
            }
    
    return None


def quality_threshold_strategy(insight: LearningInsight) -> Optional[Dict[str, Any]]:
    """Learning strategy for adjusting quality thresholds"""
    
    if insight.insight_type == "satisfaction_variability":
        rating_std = insight.supporting_data.get("rating_std", 0)
        
        if rating_std > 1.5:  # High variability
            return {
                "type": "quality_adjustment",
                "component_id": insight.component_id,
                "operation": insight.operation,
                "adjustments": {
                    "quality_threshold_increase": 0.1,  # Increase quality threshold
                    "validation_steps_added": ["output_consistency_check", "user_expectation_validation"],
                    "fallback_enabled": True
                },
                "reason": "Improve consistency based on variable user satisfaction"
            }
    
    return None


def error_handling_strategy(insight: LearningInsight) -> Optional[Dict[str, Any]]:
    """Learning strategy for improving error handling"""
    
    if insight.insight_type == "error_pattern":
        error_rate = insight.supporting_data.get("error_rate", 0)
        
        if error_rate > 0.1:  # More than 10% error rate
            return {
                "type": "error_handling_improvement",
                "component_id": insight.component_id,
                "operation": insight.operation,
                "adjustments": {
                    "retry_attempts_increase": 2,
                    "input_validation_enhanced": True,
                    "fallback_mechanisms": ["alternative_approach", "graceful_degradation"],
                    "error_reporting_improved": True
                },
                "reason": "Reduce error rate based on user feedback"
            }
    
    return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the feedback system
    storage = FeedbackStorage("test_feedback.db")
    collector = FeedbackCollector(storage)
    analyzer = FeedbackAnalyzer(storage)
    learning_engine = ContinuousLearningEngine(storage, analyzer)
    
    # Register learning strategies
    learning_engine.register_learning_strategy("parameter_tuning", parameter_tuning_strategy)
    learning_engine.register_learning_strategy("quality_threshold", quality_threshold_strategy)
    learning_engine.register_learning_strategy("error_handling", error_handling_strategy)
    
    # Simulate some task executions and feedback
    print("Simulating task executions and user feedback...")
    
    # Create sample task executions
    for i in range(10):
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            component_id="research_agent",
            component_type=ComponentType.AGENT,
            operation="web_search",
            input_data={"query": f"test query {i}"},
            output_data={"results": [f"result {j}" for j in range(5)]},
            outcome=TaskOutcome.SUCCESS if i < 8 else TaskOutcome.FAILURE,
            execution_time=np.random.uniform(1.0, 10.0),
            timestamp=time.time() - (i * 3600),  # Spread over time
            user_id=f"user_{i % 3}"
        )
        collector.register_task_execution(execution)
        
        # Add some feedback
        if i % 2 == 0:  # 50% feedback rate
            feedback = collector.collect_feedback(
                task_execution_id=execution.id,
                user_id=execution.user_id,
                feedback_type=FeedbackType.RATING,
                rating=np.random.uniform(2.0, 5.0),
                text_feedback=f"Test feedback for execution {i}"
            )
    
    # Analyze performance
    print("\nAnalyzing component performance...")
    profile = analyzer.analyze_component_performance("research_agent")
    print(f"Average Rating: {profile.average_rating:.2f}")
    print(f"Satisfaction Rate: {profile.satisfaction_rate:.1f}%")
    print(f"Improvement Trend: {profile.improvement_trend:.1f}%")
    print(f"Recommendations: {profile.recommendations}")
    
    # Generate insights
    print("\nGenerating learning insights...")
    insights = analyzer.generate_learning_insights()
    for insight in insights:
        print(f"- {insight.insight_type}: {insight.description}")
    
    # Apply learning
    print("\nApplying continuous learning...")
    adaptations = learning_engine.apply_learning()
    for adaptation in adaptations:
        print(f"- Applied {adaptation['strategy']}: {adaptation['adaptation']['type']}")
    
    print("\nFeedback system demonstration complete!")