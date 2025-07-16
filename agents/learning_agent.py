"""
Learning Agent - Reviews historical task data to identify performance bottlenecks,
recommend agent behavior optimizations, and suggest creation of new tools or sub-agents
"""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import sqlite3
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import pickle

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent_framework import BaseAgent, AgentCapability, MessageBus, Message, MessageType, AgentStatus


class LearningType(Enum):
    """Types of learning the agent can perform"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BEHAVIOR_PATTERN_ANALYSIS = "behavior_pattern_analysis"
    TOOL_RECOMMENDATION = "tool_recommendation"
    AGENT_RECOMMENDATION = "agent_recommendation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    RESOURCE_OPTIMIZATION = "resource_optimization"


class OptimizationType(Enum):
    """Types of optimizations that can be recommended"""
    AGENT_PARAMETER_TUNING = "agent_parameter_tuning"
    TASK_ROUTING_OPTIMIZATION = "task_routing_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"
    CACHING_STRATEGY = "caching_strategy"
    LOAD_BALANCING = "load_balancing"
    WORKFLOW_RESTRUCTURING = "workflow_restructuring"


@dataclass
class TaskPerformanceData:
    """Represents performance data for a task execution"""
    task_id: str
    agent_id: str
    task_type: str
    start_time: datetime
    end_time: datetime
    execution_time: float
    success: bool
    error_message: Optional[str]
    resource_usage: Dict[str, float]
    input_size: int
    output_size: int
    complexity_score: float
    user_satisfaction: Optional[float]
    context: Dict[str, Any]


@dataclass
class PerformanceBottleneck:
    """Represents an identified performance bottleneck"""
    id: str
    type: str
    component: str
    description: str
    impact_score: float
    frequency: int
    avg_delay: float
    affected_tasks: List[str]
    root_causes: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class OptimizationRecommendation:
    """Represents a recommendation for system optimization"""
    id: str
    type: OptimizationType
    target_component: str
    description: str
    expected_improvement: Dict[str, float]
    implementation_effort: str
    risk_level: str
    prerequisites: List[str]
    success_probability: float
    evidence: List[str]


@dataclass
class ToolRecommendation:
    """Represents a recommendation for a new tool"""
    id: str
    name: str
    description: str
    functionality: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    use_cases: List[str]
    frequency_need: int
    complexity_reduction: float
    time_savings: float
    implementation_priority: str


@dataclass
class AgentRecommendation:
    """Represents a recommendation for a new agent"""
    id: str
    name: str
    description: str
    capabilities: List[str]
    specialization: str
    workload_justification: str
    expected_performance_gain: float
    resource_requirements: Dict[str, Any]
    integration_complexity: str


class PerformanceDatabase:
    """Database for storing and analyzing performance data"""
    
    def __init__(self, db_path: str = "learning_agent.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the performance tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_performance (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                resource_usage TEXT,
                input_size INTEGER,
                output_size INTEGER,
                complexity_score REAL,
                user_satisfaction REAL,
                context TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_bottlenecks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                component TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_score REAL NOT NULL,
                frequency INTEGER NOT NULL,
                avg_delay REAL NOT NULL,
                affected_tasks TEXT,
                root_causes TEXT,
                recommendations TEXT,
                confidence REAL NOT NULL,
                identified_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_recommendations (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                target_component TEXT NOT NULL,
                description TEXT NOT NULL,
                expected_improvement TEXT,
                implementation_effort TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                prerequisites TEXT,
                success_probability REAL NOT NULL,
                evidence TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_recommendations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                functionality TEXT NOT NULL,
                input_schema TEXT,
                output_schema TEXT,
                use_cases TEXT,
                frequency_need INTEGER NOT NULL,
                complexity_reduction REAL NOT NULL,
                time_savings REAL NOT NULL,
                implementation_priority TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_recommendations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                capabilities TEXT,
                specialization TEXT NOT NULL,
                workload_justification TEXT NOT NULL,
                expected_performance_gain REAL NOT NULL,
                resource_requirements TEXT,
                integration_complexity TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                data_points INTEGER NOT NULL,
                confidence REAL NOT NULL,
                actionable BOOLEAN NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_task_performance(self, performance_data: TaskPerformanceData):
        """Store task performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO task_performance 
            (task_id, agent_id, task_type, start_time, end_time, execution_time,
             success, error_message, resource_usage, input_size, output_size,
             complexity_score, user_satisfaction, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            performance_data.task_id,
            performance_data.agent_id,
            performance_data.task_type,
            performance_data.start_time.isoformat(),
            performance_data.end_time.isoformat(),
            performance_data.execution_time,
            performance_data.success,
            performance_data.error_message,
            json.dumps(performance_data.resource_usage),
            performance_data.input_size,
            performance_data.output_size,
            performance_data.complexity_score,
            performance_data.user_satisfaction,
            json.dumps(performance_data.context)
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_data(self, time_range_days: int = 30, 
                           agent_id: Optional[str] = None, task_type: Optional[str] = None) -> List[TaskPerformanceData]:
        """Retrieve performance data for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM task_performance 
            WHERE start_time > datetime('now', '-{} days')
        """.format(time_range_days)
        
        params = []
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        
        query += " ORDER BY start_time DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        performance_data = []
        for row in rows:
            data = TaskPerformanceData(
                task_id=row[0],
                agent_id=row[1],
                task_type=row[2],
                start_time=datetime.fromisoformat(row[3]),
                end_time=datetime.fromisoformat(row[4]),
                execution_time=row[5],
                success=bool(row[6]),
                error_message=row[7],
                resource_usage=json.loads(row[8]) if row[8] else {},
                input_size=row[9] or 0,
                output_size=row[10] or 0,
                complexity_score=row[11] or 0.0,
                user_satisfaction=row[12],
                context=json.loads(row[13]) if row[13] else {}
            )
            performance_data.append(data)
        
        conn.close()
        return performance_data


class PatternAnalyzer:
    """Analyzes patterns in task execution and system behavior"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.analysis_cache_ttl = 3600  # 1 hour
    
    def analyze_performance_patterns(self, performance_data: List[TaskPerformanceData]) -> Dict[str, Any]:
        """Analyze performance patterns in the data"""
        if not performance_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for data in performance_data:
            df_data.append({
                'task_id': data.task_id,
                'agent_id': data.agent_id,
                'task_type': data.task_type,
                'execution_time': data.execution_time,
                'success': data.success,
                'input_size': data.input_size,
                'output_size': data.output_size,
                'complexity_score': data.complexity_score,
                'hour': data.start_time.hour,
                'day_of_week': data.start_time.weekday(),
                'user_satisfaction': data.user_satisfaction or 0.0
            })
        
        df = pd.DataFrame(df_data)
        
        patterns = {
            'execution_time_patterns': self._analyze_execution_time_patterns(df),
            'success_rate_patterns': self._analyze_success_rate_patterns(df),
            'agent_performance_patterns': self._analyze_agent_performance_patterns(df),
            'task_type_patterns': self._analyze_task_type_patterns(df),
            'temporal_patterns': self._analyze_temporal_patterns(df),
            'complexity_patterns': self._analyze_complexity_patterns(df),
            'satisfaction_patterns': self._analyze_satisfaction_patterns(df)
        }
        
        return patterns
    
    def _analyze_execution_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze execution time patterns"""
        if df.empty:
            return {}
        
        patterns = {
            'overall_stats': {
                'mean': df['execution_time'].mean(),
                'median': df['execution_time'].median(),
                'std': df['execution_time'].std(),
                'min': df['execution_time'].min(),
                'max': df['execution_time'].max()
            },
            'by_agent': df.groupby('agent_id')['execution_time'].agg(['mean', 'std', 'count']).to_dict('index'),
            'by_task_type': df.groupby('task_type')['execution_time'].agg(['mean', 'std', 'count']).to_dict('index'),
            'outliers': self._identify_execution_time_outliers(df)
        }
        
        return patterns
    
    def _analyze_success_rate_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze success rate patterns"""
        if df.empty:
            return {}
        
        patterns = {
            'overall_success_rate': df['success'].mean(),
            'by_agent': df.groupby('agent_id')['success'].agg(['mean', 'count']).to_dict('index'),
            'by_task_type': df.groupby('task_type')['success'].agg(['mean', 'count']).to_dict('index'),
            'by_hour': df.groupby('hour')['success'].mean().to_dict(),
            'by_day_of_week': df.groupby('day_of_week')['success'].mean().to_dict()
        }
        
        return patterns
    
    def _analyze_agent_performance_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual agent performance patterns"""
        if df.empty:
            return {}
        
        agent_patterns = {}
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            agent_patterns[agent_id] = {
                'task_count': len(agent_df),
                'success_rate': agent_df['success'].mean(),
                'avg_execution_time': agent_df['execution_time'].mean(),
                'task_types': agent_df['task_type'].value_counts().to_dict(),
                'performance_trend': self._calculate_performance_trend(agent_df),
                'efficiency_score': self._calculate_efficiency_score(agent_df)
            }
        
        return agent_patterns
    
    def _analyze_task_type_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns by task type"""
        if df.empty:
            return {}
        
        task_patterns = {}
        
        for task_type in df['task_type'].unique():
            task_df = df[df['task_type'] == task_type]
            
            task_patterns[task_type] = {
                'frequency': len(task_df),
                'success_rate': task_df['success'].mean(),
                'avg_execution_time': task_df['execution_time'].mean(),
                'complexity_distribution': self._analyze_complexity_distribution(task_df),
                'agent_performance': task_df.groupby('agent_id')['execution_time'].mean().to_dict()
            }
        
        return task_patterns
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in task execution"""
        if df.empty:
            return {}
        
        patterns = {
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict(),
            'daily_distribution': df['day_of_week'].value_counts().sort_index().to_dict(),
            'peak_hours': self._identify_peak_hours(df),
            'performance_by_time': {
                'hourly_performance': df.groupby('hour')['execution_time'].mean().to_dict(),
                'daily_performance': df.groupby('day_of_week')['execution_time'].mean().to_dict()
            }
        }
        
        return patterns
    
    def _analyze_complexity_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze complexity vs performance patterns"""
        if df.empty or 'complexity_score' not in df.columns:
            return {}
        
        # Filter out zero complexity scores
        complex_df = df[df['complexity_score'] > 0]
        
        if complex_df.empty:
            return {}
        
        patterns = {
            'complexity_vs_time_correlation': complex_df['complexity_score'].corr(complex_df['execution_time']),
            'complexity_vs_success_correlation': complex_df['complexity_score'].corr(complex_df['success']),
            'complexity_distribution': {
                'low': len(complex_df[complex_df['complexity_score'] < 0.3]),
                'medium': len(complex_df[(complex_df['complexity_score'] >= 0.3) & (complex_df['complexity_score'] < 0.7)]),
                'high': len(complex_df[complex_df['complexity_score'] >= 0.7])
            }
        }
        
        return patterns
    
    def _analyze_satisfaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user satisfaction patterns"""
        if df.empty or 'user_satisfaction' not in df.columns:
            return {}
        
        # Filter out null satisfaction scores
        satisfaction_df = df[df['user_satisfaction'].notna()]
        
        if satisfaction_df.empty:
            return {}
        
        patterns = {
            'overall_satisfaction': satisfaction_df['user_satisfaction'].mean(),
            'satisfaction_vs_time_correlation': satisfaction_df['user_satisfaction'].corr(satisfaction_df['execution_time']),
            'satisfaction_by_agent': satisfaction_df.groupby('agent_id')['user_satisfaction'].mean().to_dict(),
            'satisfaction_by_task_type': satisfaction_df.groupby('task_type')['user_satisfaction'].mean().to_dict(),
            'low_satisfaction_tasks': satisfaction_df[satisfaction_df['user_satisfaction'] < 0.5]['task_type'].value_counts().to_dict()
        }
        
        return patterns
    
    def _identify_execution_time_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify execution time outliers"""
        if df.empty:
            return []
        
        Q1 = df['execution_time'].quantile(0.25)
        Q3 = df['execution_time'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['execution_time'] < lower_bound) | (df['execution_time'] > upper_bound)]
        
        # Convert to records and ensure all keys are strings
        records = outliers[['task_id', 'agent_id', 'task_type', 'execution_time']].to_dict('records')
        return [{str(k): v for k, v in record.items()} for record in records]
    
    def _calculate_performance_trend(self, agent_df: pd.DataFrame) -> str:
        """Calculate performance trend for an agent"""
        if len(agent_df) < 5:
            return "insufficient_data"
        
        # Sort by task_id (assuming chronological order)
        sorted_df = agent_df.sort_values('task_id')
        
        # Calculate trend in execution time
        x = np.arange(len(sorted_df), dtype=np.float64)
        y = sorted_df['execution_time'].values.astype(np.float64)
        
        # Handle NaN values
        if np.any(np.isnan(y)):
            return "insufficient_data"
        
        # Simple linear regression
        try:
            slope = np.polyfit(x, y, 1)[0]
        except (ValueError, np.linalg.LinAlgError):
            return "insufficient_data"
        
        if slope < -0.1:
            return "improving"
        elif slope > 0.1:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_efficiency_score(self, agent_df: pd.DataFrame) -> float:
        """Calculate efficiency score for an agent"""
        if agent_df.empty:
            return 0.0
        
        success_rate = agent_df['success'].mean()
        avg_time = agent_df['execution_time'].mean()
        
        # Normalize execution time (assuming 60 seconds is baseline)
        time_score = max(0, 1 - (avg_time / 60))
        
        # Combine success rate and time efficiency
        efficiency = (success_rate * 0.7) + (time_score * 0.3)
        
        return min(1.0, max(0.0, efficiency))
    
    def _analyze_complexity_distribution(self, task_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze complexity distribution for a task type"""
        if task_df.empty or 'complexity_score' not in task_df.columns:
            return {'low': 0, 'medium': 0, 'high': 0}
        
        complex_df = task_df[task_df['complexity_score'] > 0]
        
        return {
            'low': len(complex_df[complex_df['complexity_score'] < 0.3]),
            'medium': len(complex_df[(complex_df['complexity_score'] >= 0.3) & (complex_df['complexity_score'] < 0.7)]),
            'high': len(complex_df[complex_df['complexity_score'] >= 0.7])
        }
    
    def _identify_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Identify peak usage hours"""
        if df.empty:
            return []
        
        hourly_counts = df['hour'].value_counts()
        mean_count = hourly_counts.mean()
        std_count = hourly_counts.std()
        
        threshold = mean_count + std_count
        # Convert to numeric for comparison and handle NaN values
        peak_hours = hourly_counts[hourly_counts.astype(float) > threshold].index.tolist()
        
        return sorted(peak_hours)


class RecommendationEngine:
    """Generates recommendations based on analysis results"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.recommendation_templates = self._load_recommendation_templates()
    
    def generate_optimization_recommendations(self, patterns: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on patterns"""
        recommendations = []
        
        # Analyze agent performance patterns
        agent_patterns = patterns.get('agent_performance_patterns', {})
        for agent_id, agent_data in agent_patterns.items():
            if agent_data['efficiency_score'] < 0.6:
                recommendations.append(self._create_agent_optimization_recommendation(agent_id, agent_data))
        
        # Analyze task type patterns
        task_patterns = patterns.get('task_type_patterns', {})
        for task_type, task_data in task_patterns.items():
            if task_data['avg_execution_time'] > 60:  # Tasks taking more than 1 minute
                recommendations.append(self._create_task_optimization_recommendation(task_type, task_data))
        
        # Analyze temporal patterns
        temporal_patterns = patterns.get('temporal_patterns', {})
        if temporal_patterns.get('peak_hours'):
            recommendations.append(self._create_load_balancing_recommendation(temporal_patterns))
        
        return recommendations
    
    def generate_tool_recommendations(self, patterns: Dict[str, Any]) -> List[ToolRecommendation]:
        """Generate tool recommendations based on analysis"""
        recommendations = []
        
        # Analyze frequently failing task types
        task_patterns = patterns.get('task_type_patterns', {})
        for task_type, task_data in task_patterns.items():
            if task_data['success_rate'] < 0.8 and task_data['frequency'] > 10:
                recommendations.append(self._create_tool_recommendation_for_failing_tasks(task_type, task_data))
        
        # Analyze high complexity tasks
        complexity_patterns = patterns.get('complexity_patterns', {})
        if complexity_patterns.get('complexity_distribution', {}).get('high', 0) > 20:
            recommendations.append(self._create_complexity_reduction_tool_recommendation(complexity_patterns))
        
        return recommendations
    
    def generate_agent_recommendations(self, patterns: Dict[str, Any]) -> List[AgentRecommendation]:
        """Generate agent recommendations based on workload analysis"""
        recommendations = []
        
        # Analyze agent workload distribution
        agent_patterns = patterns.get('agent_performance_patterns', {})
        task_patterns = patterns.get('task_type_patterns', {})
        
        # Identify overloaded agents
        overloaded_agents = []
        for agent_id, agent_data in agent_patterns.items():
            if agent_data['task_count'] > 100 and agent_data['efficiency_score'] < 0.7:
                overloaded_agents.append((agent_id, agent_data))
        
        if overloaded_agents:
            recommendations.append(self._create_load_distribution_agent_recommendation(overloaded_agents))
        
        # Identify specialized task types that could benefit from dedicated agents
        for task_type, task_data in task_patterns.items():
            if (task_data['frequency'] > 50 and 
                task_data['avg_execution_time'] > 30 and
                len(task_data['agent_performance']) > 3):
                recommendations.append(self._create_specialized_agent_recommendation(task_type, task_data))
        
        return recommendations
    
    def _create_agent_optimization_recommendation(self, agent_id: str, agent_data: Dict[str, Any]) -> OptimizationRecommendation:
        """Create optimization recommendation for underperforming agent"""
        return OptimizationRecommendation(
            id=str(uuid.uuid4()),
            type=OptimizationType.AGENT_PARAMETER_TUNING,
            target_component=agent_id,
            description=f"Optimize parameters for agent {agent_id} with efficiency score {agent_data['efficiency_score']:.2f}",
            expected_improvement={
                'efficiency_increase': 0.2,
                'execution_time_reduction': 0.15
            },
            implementation_effort="medium",
            risk_level="low",
            prerequisites=["agent_configuration_access"],
            success_probability=0.75,
            evidence=[
                f"Current efficiency score: {agent_data['efficiency_score']:.2f}",
                f"Success rate: {agent_data['success_rate']:.2f}",
                f"Average execution time: {agent_data['avg_execution_time']:.2f}s"
            ]
        )
    
    def _create_task_optimization_recommendation(self, task_type: str, task_data: Dict[str, Any]) -> OptimizationRecommendation:
        """Create optimization recommendation for slow task type"""
        return OptimizationRecommendation(
            id=str(uuid.uuid4()),
            type=OptimizationType.WORKFLOW_RESTRUCTURING,
            target_component=task_type,
            description=f"Optimize workflow for {task_type} tasks with average execution time {task_data['avg_execution_time']:.2f}s",
            expected_improvement={
                'execution_time_reduction': 0.3,
                'success_rate_increase': 0.1
            },
            implementation_effort="high",
            risk_level="medium",
            prerequisites=["workflow_analysis", "task_decomposition"],
            success_probability=0.65,
            evidence=[
                f"Average execution time: {task_data['avg_execution_time']:.2f}s",
                f"Task frequency: {task_data['frequency']}",
                f"Success rate: {task_data['success_rate']:.2f}"
            ]
        )
    
    def _create_load_balancing_recommendation(self, temporal_patterns: Dict[str, Any]) -> OptimizationRecommendation:
        """Create load balancing recommendation based on temporal patterns"""
        peak_hours = temporal_patterns.get('peak_hours', [])
        
        return OptimizationRecommendation(
            id=str(uuid.uuid4()),
            type=OptimizationType.LOAD_BALANCING,
            target_component="system_orchestrator",
            description=f"Implement load balancing for peak hours: {peak_hours}",
            expected_improvement={
                'response_time_reduction': 0.25,
                'system_stability_increase': 0.2
            },
            implementation_effort="high",
            risk_level="medium",
            prerequisites=["load_balancer_implementation", "resource_scaling"],
            success_probability=0.8,
            evidence=[
                f"Peak hours identified: {peak_hours}",
                f"Hourly distribution variance: high"
            ]
        )
    
    def _create_tool_recommendation_for_failing_tasks(self, task_type: str, task_data: Dict[str, Any]) -> ToolRecommendation:
        """Create tool recommendation for frequently failing tasks"""
        return ToolRecommendation(
            id=str(uuid.uuid4()),
            name=f"{task_type}_reliability_tool",
            description=f"Tool to improve reliability of {task_type} tasks",
            functionality=f"Provides enhanced error handling, retry logic, and validation for {task_type} operations",
            input_schema={"task_parameters": "object", "retry_config": "object"},
            output_schema={"result": "object", "reliability_metrics": "object"},
            use_cases=[f"Reliable {task_type} execution", "Error recovery", "Performance monitoring"],
            frequency_need=task_data['frequency'],
            complexity_reduction=0.3,
            time_savings=task_data['avg_execution_time'] * 0.2,
            implementation_priority="high"
        )
    
    def _create_complexity_reduction_tool_recommendation(self, complexity_patterns: Dict[str, Any]) -> ToolRecommendation:
        """Create tool recommendation for complexity reduction"""
        return ToolRecommendation(
            id=str(uuid.uuid4()),
            name="complexity_analyzer_tool",
            description="Tool to analyze and reduce task complexity",
            functionality="Analyzes task complexity and suggests decomposition strategies",
            input_schema={"task_description": "string", "complexity_metrics": "object"},
            output_schema={"complexity_score": "number", "decomposition_suggestions": "array"},
            use_cases=["Task complexity analysis", "Workflow optimization", "Performance prediction"],
            frequency_need=complexity_patterns.get('complexity_distribution', {}).get('high', 0),
            complexity_reduction=0.4,
            time_savings=30.0,
            implementation_priority="medium"
        )
    
    def _create_load_distribution_agent_recommendation(self, overloaded_agents: List[Tuple[str, Dict[str, Any]]]) -> AgentRecommendation:
        """Create agent recommendation for load distribution"""
        agent_names = [agent[0] for agent in overloaded_agents]
        
        return AgentRecommendation(
            id=str(uuid.uuid4()),
            name="load_balancer_agent",
            description="Agent to distribute load among overloaded agents",
            capabilities=["task_routing", "load_monitoring", "dynamic_scaling"],
            specialization="Load balancing and task distribution",
            workload_justification=f"Agents {agent_names} are overloaded with efficiency scores below 0.7",
            expected_performance_gain=0.3,
            resource_requirements={"cpu": "medium", "memory": "low", "network": "high"},
            integration_complexity="medium"
        )
    
    def _create_specialized_agent_recommendation(self, task_type: str, task_data: Dict[str, Any]) -> AgentRecommendation:
        """Create recommendation for specialized agent"""
        return AgentRecommendation(
            id=str(uuid.uuid4()),
            name=f"{task_type}_specialist_agent",
            description=f"Specialized agent for {task_type} tasks",
            capabilities=[f"{task_type}_processing", f"{task_type}_optimization"],
            specialization=f"Optimized for {task_type} task execution",
            workload_justification=f"High frequency ({task_data['frequency']}) and execution time ({task_data['avg_execution_time']:.2f}s) for {task_type}",
            expected_performance_gain=0.4,
            resource_requirements={"cpu": "high", "memory": "medium", "network": "medium"},
            integration_complexity="low"
        )
    
    def _load_recommendation_templates(self) -> Dict[str, Any]:
        """Load recommendation templates"""
        return {
            "performance_optimization": {
                "threshold_efficiency": 0.6,
                "threshold_execution_time": 60,
                "threshold_success_rate": 0.8
            },
            "tool_creation": {
                "min_frequency": 10,
                "min_complexity_reduction": 0.2,
                "min_time_savings": 5.0
            },
            "agent_creation": {
                "min_task_frequency": 50,
                "min_execution_time": 30,
                "min_agent_count": 3
            }
        }


class LearningAgent(BaseAgent):
    """
    Learning agent that analyzes historical data and provides optimization recommendations
    """
    
    def __init__(self, message_bus: MessageBus, db_path: str = "learning_agent.db"):
        capabilities = [
            AgentCapability(
                name="performance_analysis",
                description="Analyze system performance patterns and bottlenecks",
                input_schema={
                    "time_range_days": "integer",
                    "component_filter": "string",
                    "analysis_type": "string"
                },
                output_schema={
                    "patterns": "object",
                    "bottlenecks": "array",
                    "insights": "array"
                },
                estimated_duration=30.0,
                max_concurrent=3
            ),
            AgentCapability(
                name="optimization_recommendations",
                description="Generate optimization recommendations",
                input_schema={
                    "target_component": "string",
                    "optimization_type": "string"
                },
                output_schema={
                    "recommendations": "array",
                    "priority_ranking": "array"
                },
                estimated_duration=20.0,
                max_concurrent=5
            ),
            AgentCapability(
                name="tool_recommendations",
                description="Recommend new tools based on usage patterns",
                input_schema={
                    "analysis_scope": "string",
                    "priority_threshold": "number"
                },
                output_schema={
                    "tool_recommendations": "array",
                    "implementation_roadmap": "object"
                },
                estimated_duration=25.0,
                max_concurrent=3
            ),
            AgentCapability(
                name="agent_recommendations",
                description="Recommend new agents based on workload analysis",
                input_schema={
                    "workload_threshold": "number",
                    "specialization_analysis": "boolean"
                },
                output_schema={
                    "agent_recommendations": "array",
                    "integration_plan": "object"
                },
                estimated_duration=35.0,
                max_concurrent=2
            ),
            AgentCapability(
                name="continuous_learning",
                description="Continuously learn from system behavior",
                input_schema={
                    "learning_mode": "string",
                    "feedback_data": "object"
                },
                output_schema={
                    "learning_insights": "array",
                    "model_updates": "object"
                },
                estimated_duration=60.0,
                max_concurrent=1
            )
        ]
        
        super().__init__("learning_agent", capabilities, message_bus)
        
        self.db = PerformanceDatabase(db_path)
        self.pattern_analyzer = PatternAnalyzer()
        self.recommendation_engine = RecommendationEngine(self.db)
        self.learning_active = True
        
        # Start continuous learning process
        self._start_continuous_learning()
        
        # Subscribe to task completion messages
        self.message_bus.subscribe(MessageType.TASK_RESPONSE, self._handle_task_completion)
    
    def _start_continuous_learning(self):
        """Start continuous learning background process"""
        def learning_loop():
            while self.learning_active:
                try:
                    self._perform_periodic_analysis()
                    time.sleep(3600)  # Analyze every hour
                except Exception as e:
                    logging.error(f"Continuous learning error: {e}")
                    time.sleep(1800)  # Wait 30 minutes on error
        
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
    
    def _perform_periodic_analysis(self):
        """Perform periodic analysis of system performance"""
        try:
            # Get recent performance data
            performance_data = self.db.get_performance_data(time_range_days=1)
            
            if len(performance_data) < 10:
                return  # Not enough data for meaningful analysis
            
            # Analyze patterns
            patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
            
            # Generate insights
            insights = self._generate_insights(patterns)
            
            # Store insights
            for insight in insights:
                self._store_insight(insight)
            
            # Generate recommendations if significant patterns found
            if self._should_generate_recommendations(patterns):
                self._generate_and_store_recommendations(patterns)
            
        except Exception as e:
            logging.error(f"Periodic analysis error: {e}")
    
    def _handle_task_completion(self, message: Message):
        """Handle task completion messages to collect performance data"""
        try:
            payload = message.payload
            result_data = payload.get('result', {})
            
            if not result_data:
                return
            
            # Extract performance data
            performance_data = TaskPerformanceData(
                task_id=payload.get('task_id', str(uuid.uuid4())),
                agent_id=message.sender,
                task_type=result_data.get('task_type', 'unknown'),
                start_time=datetime.fromtimestamp(message.timestamp - result_data.get('execution_time', 0)),
                end_time=datetime.fromtimestamp(message.timestamp),
                execution_time=result_data.get('execution_time', 0),
                success=result_data.get('success', False),
                error_message=result_data.get('error'),
                resource_usage=result_data.get('resource_usage', {}),
                input_size=result_data.get('input_size', 0),
                output_size=result_data.get('output_size', 0),
                complexity_score=result_data.get('complexity_score', 0.0),
                user_satisfaction=result_data.get('user_satisfaction'),
                context=result_data.get('context', {})
            )
            
            # Store performance data
            self.db.store_task_performance(performance_data)
            
        except Exception as e:
            logging.error(f"Error handling task completion: {e}")
    
    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute learning agent capabilities"""
        try:
            if capability_name == "performance_analysis":
                return self._analyze_performance(payload)
            elif capability_name == "optimization_recommendations":
                return self._generate_optimization_recommendations(payload)
            elif capability_name == "tool_recommendations":
                return self._generate_tool_recommendations(payload)
            elif capability_name == "agent_recommendations":
                return self._generate_agent_recommendations(payload)
            elif capability_name == "continuous_learning":
                return self._continuous_learning(payload)
            else:
                raise ValueError(f"Unknown capability: {capability_name}")
        except Exception as e:
            logging.error(f"Error executing capability {capability_name}: {e}")
            raise
    
    def _analyze_performance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance"""
        time_range_days = payload.get('time_range_days', 30)
        component_filter = payload.get('component_filter')
        analysis_type = payload.get('analysis_type', 'comprehensive')
        
        # Get performance data
        performance_data = self.db.get_performance_data(
            time_range_days=time_range_days,
            agent_id=component_filter if analysis_type == 'agent' else None,
            task_type=component_filter if analysis_type == 'task_type' else None
        )
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(patterns)
        
        # Generate insights
        insights = self._generate_insights(patterns)
        
        return {
            'patterns': patterns,
            'bottlenecks': [asdict(b) for b in bottlenecks],
            'insights': insights
        }
    
    def _generate_optimization_recommendations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        target_component = payload.get('target_component')
        optimization_type = payload.get('optimization_type', 'all')
        
        # Get recent performance data
        performance_data = self.db.get_performance_data(time_range_days=30)
        patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_optimization_recommendations(patterns)
        
        # Filter by target component if specified
        if target_component:
            recommendations = [r for r in recommendations if target_component in r.target_component]
        
        # Filter by optimization type if specified
        if optimization_type != 'all':
            recommendations = [r for r in recommendations if r.type.value == optimization_type]
        
        # Rank by priority
        priority_ranking = self._rank_recommendations(recommendations)
        
        return {
            'recommendations': [asdict(r) for r in recommendations],
            'priority_ranking': priority_ranking
        }
    
    def _generate_tool_recommendations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tool recommendations"""
        analysis_scope = payload.get('analysis_scope', 'system')
        priority_threshold = payload.get('priority_threshold', 0.5)
        
        # Get performance data
        performance_data = self.db.get_performance_data(time_range_days=30)
        patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
        
        # Generate tool recommendations
        tool_recommendations = self.recommendation_engine.generate_tool_recommendations(patterns)
        
        # Filter by priority threshold
        filtered_recommendations = [
            r for r in tool_recommendations 
            if self._calculate_tool_priority_score(r) >= priority_threshold
        ]
        
        # Create implementation roadmap
        roadmap = self._create_tool_implementation_roadmap(filtered_recommendations)
        
        return {
            'tool_recommendations': [asdict(r) for r in filtered_recommendations],
            'implementation_roadmap': roadmap
        }
    
    def _generate_agent_recommendations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent recommendations"""
        workload_threshold = payload.get('workload_threshold', 100)
        specialization_analysis = payload.get('specialization_analysis', True)
        
        # Get performance data
        performance_data = self.db.get_performance_data(time_range_days=30)
        patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
        
        # Generate agent recommendations
        agent_recommendations = self.recommendation_engine.generate_agent_recommendations(patterns)
        
        # Create integration plan
        integration_plan = self._create_agent_integration_plan(agent_recommendations)
        
        return {
            'agent_recommendations': [asdict(r) for r in agent_recommendations],
            'integration_plan': integration_plan
        }
    
    def _continuous_learning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform continuous learning"""
        learning_mode = payload.get('learning_mode', 'adaptive')
        feedback_data = payload.get('feedback_data', {})
        
        # Get recent data for learning
        performance_data = self.db.get_performance_data(time_range_days=7)
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_performance_patterns(performance_data)
        
        # Generate learning insights
        insights = self._generate_learning_insights(patterns, feedback_data)
        
        # Update internal models (placeholder for ML model updates)
        model_updates = self._update_learning_models(patterns, feedback_data)
        
        return {
            'learning_insights': insights,
            'model_updates': model_updates
        }
    
    def _identify_bottlenecks(self, patterns: Dict[str, Any]) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks from patterns"""
        bottlenecks = []
        
        # Check execution time patterns
        execution_patterns = patterns.get('execution_time_patterns', {})
        outliers = execution_patterns.get('outliers', [])
        
        if outliers:
            bottleneck = PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="execution_time_outlier",
                component="system",
                description=f"Found {len(outliers)} execution time outliers",
                impact_score=len(outliers) / 100.0,
                frequency=len(outliers),
                avg_delay=float(np.mean([o['execution_time'] for o in outliers])),
                affected_tasks=[o['task_id'] for o in outliers],
                root_causes=["resource_contention", "inefficient_algorithms"],
                recommendations=["optimize_algorithms", "increase_resources"],
                confidence=0.8
            )
            bottlenecks.append(bottleneck)
        
        # Check agent performance patterns
        agent_patterns = patterns.get('agent_performance_patterns', {})
        for agent_id, agent_data in agent_patterns.items():
            if agent_data['efficiency_score'] < 0.5:
                bottleneck = PerformanceBottleneck(
                    id=str(uuid.uuid4()),
                    type="agent_performance",
                    component=agent_id,
                    description=f"Agent {agent_id} has low efficiency score: {agent_data['efficiency_score']:.2f}",
                    impact_score=1.0 - agent_data['efficiency_score'],
                    frequency=agent_data['task_count'],
                    avg_delay=float(agent_data['avg_execution_time']),
                    affected_tasks=[],
                    root_causes=["configuration_issues", "resource_limitations"],
                    recommendations=["tune_parameters", "allocate_more_resources"],
                    confidence=0.9
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _generate_insights(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from patterns"""
        insights = []
        
        # Overall system insights
        execution_patterns = patterns.get('execution_time_patterns', {})
        if execution_patterns.get('overall_stats'):
            stats = execution_patterns['overall_stats']
            insights.append({
                'type': 'system_performance',
                'description': f"Average execution time: {stats['mean']:.2f}s, Standard deviation: {stats['std']:.2f}s",
                'actionable': stats['std'] > stats['mean'] * 0.5,
                'confidence': 0.9
            })
        
        # Agent performance insights
        agent_patterns = patterns.get('agent_performance_patterns', {})
        if agent_patterns:
            best_agent = max(agent_patterns.items(), key=lambda x: x[1]['efficiency_score'])
            worst_agent = min(agent_patterns.items(), key=lambda x: x[1]['efficiency_score'])
            
            insights.append({
                'type': 'agent_comparison',
                'description': f"Best performing agent: {best_agent[0]} (efficiency: {best_agent[1]['efficiency_score']:.2f}), "
                             f"Worst: {worst_agent[0]} (efficiency: {worst_agent[1]['efficiency_score']:.2f})",
                'actionable': worst_agent[1]['efficiency_score'] < 0.7,
                'confidence': 0.8
            })
        
        # Temporal insights
        temporal_patterns = patterns.get('temporal_patterns', {})
        peak_hours = temporal_patterns.get('peak_hours', [])
        if peak_hours:
            insights.append({
                'type': 'temporal_pattern',
                'description': f"Peak usage hours: {peak_hours}",
                'actionable': len(peak_hours) > 3,
                'confidence': 0.7
            })
        
        return insights
    
    def _generate_learning_insights(self, patterns: Dict[str, Any], feedback_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate learning insights"""
        insights = []
        
        # Pattern evolution insights
        insights.append({
            'type': 'pattern_evolution',
            'description': 'System patterns are evolving based on usage',
            'data_points': len(patterns),
            'confidence': 0.8,
            'actionable': True
        })
        
        # Feedback incorporation insights
        if feedback_data:
            insights.append({
                'type': 'feedback_learning',
                'description': 'Incorporating user feedback into learning models',
                'data_points': len(feedback_data),
                'confidence': 0.9,
                'actionable': True
            })
        
        return insights
    
    def _update_learning_models(self, patterns: Dict[str, Any], feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update learning models (placeholder for ML model updates)"""
        return {
            'pattern_model_updated': True,
            'feedback_model_updated': bool(feedback_data),
            'update_timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0'
        }
    
    def _rank_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Rank recommendations by priority"""
        ranked = []
        
        for rec in recommendations:
            priority_score = (
                rec.success_probability * 0.4 +
                sum(rec.expected_improvement.values()) * 0.3 +
                (1.0 if rec.risk_level == 'low' else 0.5 if rec.risk_level == 'medium' else 0.2) * 0.3
            )
            
            ranked.append({
                'recommendation_id': rec.id,
                'priority_score': priority_score,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by priority score
        ranked.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Set ranks
        for i, item in enumerate(ranked):
            item['rank'] = i + 1
        
        return ranked
    
    def _calculate_tool_priority_score(self, tool_rec: ToolRecommendation) -> float:
        """Calculate priority score for tool recommendation"""
        frequency_score = min(tool_rec.frequency_need / 100.0, 1.0)
        complexity_score = tool_rec.complexity_reduction
        time_score = min(tool_rec.time_savings / 60.0, 1.0)
        
        priority_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        priority_score = priority_map.get(tool_rec.implementation_priority, 0.5)
        
        return (frequency_score * 0.3 + complexity_score * 0.3 + time_score * 0.2 + priority_score * 0.2)
    
    def _create_tool_implementation_roadmap(self, tool_recommendations: List[ToolRecommendation]) -> Dict[str, Any]:
        """Create implementation roadmap for tools"""
        roadmap = {
            'phases': [],
            'total_estimated_time': 0,
            'dependencies': [],
            'resource_requirements': {}
        }
        
        # Group by priority
        high_priority = [t for t in tool_recommendations if t.implementation_priority == 'high']
        medium_priority = [t for t in tool_recommendations if t.implementation_priority == 'medium']
        low_priority = [t for t in tool_recommendations if t.implementation_priority == 'low']
        
        if high_priority:
            roadmap['phases'].append({
                'phase': 1,
                'name': 'High Priority Tools',
                'tools': [t.name for t in high_priority],
                'estimated_time_weeks': len(high_priority) * 2
            })
        
        if medium_priority:
            roadmap['phases'].append({
                'phase': 2,
                'name': 'Medium Priority Tools',
                'tools': [t.name for t in medium_priority],
                'estimated_time_weeks': len(medium_priority) * 1.5
            })
        
        if low_priority:
            roadmap['phases'].append({
                'phase': 3,
                'name': 'Low Priority Tools',
                'tools': [t.name for t in low_priority],
                'estimated_time_weeks': len(low_priority) * 1
            })
        
        roadmap['total_estimated_time'] = sum(phase['estimated_time_weeks'] for phase in roadmap['phases'])
        
        return roadmap
    
    def _create_agent_integration_plan(self, agent_recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Create integration plan for new agents"""
        plan = {
            'integration_phases': [],
            'resource_allocation': {},
            'testing_strategy': {},
            'rollback_plan': {}
        }
        
        # Sort by integration complexity
        complexity_order = {'low': 1, 'medium': 2, 'high': 3}
        sorted_agents = sorted(agent_recommendations, key=lambda x: complexity_order.get(x.integration_complexity, 2))
        
        for i, agent_rec in enumerate(sorted_agents):
            plan['integration_phases'].append({
                'phase': i + 1,
                'agent_name': agent_rec.name,
                'complexity': agent_rec.integration_complexity,
                'estimated_time_weeks': complexity_order.get(agent_rec.integration_complexity, 2) * 2,
                'prerequisites': ['testing_environment', 'integration_testing']
            })
        
        return plan
    
    def _store_insight(self, insight: Dict[str, Any]):
        """Store learning insight in database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO learning_insights 
            (id, insight_type, description, data_points, confidence, actionable, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            insight.get('type', 'general'),
            insight.get('description', ''),
            insight.get('data_points', 0),
            insight.get('confidence', 0.0),
            insight.get('actionable', False),
            datetime.now().isoformat(),
            json.dumps(insight)
        ))
        
        conn.commit()
        conn.close()
    
    def _should_generate_recommendations(self, patterns: Dict[str, Any]) -> bool:
        """Determine if recommendations should be generated"""
        # Check if there are significant patterns worth acting on
        agent_patterns = patterns.get('agent_performance_patterns', {})
        low_efficiency_agents = sum(1 for data in agent_patterns.values() if data['efficiency_score'] < 0.7)
        
        task_patterns = patterns.get('task_type_patterns', {})
        slow_tasks = sum(1 for data in task_patterns.values() if data['avg_execution_time'] > 60)
        
        return low_efficiency_agents > 0 or slow_tasks > 0
    
    def _generate_and_store_recommendations(self, patterns: Dict[str, Any]):
        """Generate and store recommendations"""
        try:
            # Generate optimization recommendations
            opt_recommendations = self.recommendation_engine.generate_optimization_recommendations(patterns)
            for rec in opt_recommendations:
                self._store_optimization_recommendation(rec)
            
            # Generate tool recommendations
            tool_recommendations = self.recommendation_engine.generate_tool_recommendations(patterns)
            for rec in tool_recommendations:
                self._store_tool_recommendation(rec)
            
            # Generate agent recommendations
            agent_recommendations = self.recommendation_engine.generate_agent_recommendations(patterns)
            for rec in agent_recommendations:
                self._store_agent_recommendation(rec)
                
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
    
    def _store_optimization_recommendation(self, rec: OptimizationRecommendation):
        """Store optimization recommendation in database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO optimization_recommendations 
            (id, type, target_component, description, expected_improvement, 
             implementation_effort, risk_level, prerequisites, success_probability, 
             evidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec.id, rec.type.value, rec.target_component, rec.description,
            json.dumps(rec.expected_improvement), rec.implementation_effort,
            rec.risk_level, json.dumps(rec.prerequisites), rec.success_probability,
            json.dumps(rec.evidence), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_tool_recommendation(self, rec: ToolRecommendation):
        """Store tool recommendation in database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tool_recommendations 
            (id, name, description, functionality, input_schema, output_schema,
             use_cases, frequency_need, complexity_reduction, time_savings,
             implementation_priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec.id, rec.name, rec.description, rec.functionality,
            json.dumps(rec.input_schema), json.dumps(rec.output_schema),
            json.dumps(rec.use_cases), rec.frequency_need, rec.complexity_reduction,
            rec.time_savings, rec.implementation_priority, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_agent_recommendation(self, rec: AgentRecommendation):
        """Store agent recommendation in database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_recommendations 
            (id, name, description, capabilities, specialization, workload_justification,
             expected_performance_gain, resource_requirements, integration_complexity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec.id, rec.name, rec.description, json.dumps(rec.capabilities),
            rec.specialization, rec.workload_justification, rec.expected_performance_gain,
            json.dumps(rec.resource_requirements), rec.integration_complexity,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def shutdown(self):
        """Gracefully shutdown the learning agent"""
        self.learning_active = False
        super().shutdown()


# Convenience function for external integration
def learning_agent(task: str) -> Dict[str, Any]:
    """
    Main entry point for the learning agent
    Compatible with the existing agent manifest system
    """
    try:
        # Parse task to determine action
        task_lower = task.lower()
        
        if 'analyze' in task_lower or 'performance' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = LearningAgent(message_bus)
            
            result = agent.execute_capability("performance_analysis", {
                "time_range_days": 30,
                "analysis_type": "comprehensive"
            })
            
            return {
                "status": "success",
                "result": result,
                "message": "Performance analysis completed"
            }
        
        elif 'recommend' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = LearningAgent(message_bus)
            
            if 'tool' in task_lower:
                result = agent.execute_capability("tool_recommendations", {
                    "analysis_scope": "system",
                    "priority_threshold": 0.5
                })
            elif 'agent' in task_lower:
                result = agent.execute_capability("agent_recommendations", {
                    "workload_threshold": 100,
                    "specialization_analysis": True
                })
            else:
                result = agent.execute_capability("optimization_recommendations", {
                    "optimization_type": "all"
                })
            
            return {
                "status": "success",
                "result": result,
                "message": "Recommendations generated"
            }
        
        else:
            return {
                "status": "success",
                "result": {
                    "message": "Learning agent initialized and analyzing system patterns",
                    "capabilities": [
                        "performance_analysis",
                        "optimization_recommendations",
                        "tool_recommendations",
                        "agent_recommendations",
                        "continuous_learning"
                    ]
                },
                "message": "Learning agent ready"
            }
            
    except Exception as e:
        return {
            "status": "failure",
            "result": None,
            "message": f"Learning agent error: {str(e)}"
        }


if __name__ == "__main__":
    # Test the learning agent
    logging.basicConfig(level=logging.INFO)
    
    from core.agent_framework import MessageBus
    message_bus = MessageBus()
    
    # Create learning agent
    learning_agent_instance = LearningAgent(message_bus)
    
    # Test performance analysis
    test_result = learning_agent_instance.execute_capability("performance_analysis", {
        "time_range_days": 7,
        "analysis_type": "comprehensive"
    })
    
    print("Performance Analysis Test:")
    print(json.dumps(test_result, indent=2, default=str))
    
    # Test optimization recommendations
    opt_result = learning_agent_instance.execute_capability("optimization_recommendations", {
        "optimization_type": "all"
    })
    
    print("\nOptimization Recommendations Test:")
    print(json.dumps(opt_result, indent=2, default=str))
    
    # Cleanup
    learning_agent_instance.shutdown()