"""
Comprehensive test and demonstration of the User Feedback Loop System
Shows how the system collects feedback and applies continuous learning.
"""

import time
import random
import logging
import threading
from typing import Dict, List, Any
import numpy as np

# Add the core directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from feedback_system import (
    FeedbackStorage, FeedbackCollector, FeedbackAnalyzer, ContinuousLearningEngine,
    TaskExecution, UserFeedback, ComponentType, TaskOutcome, FeedbackType
)
from feedback_integration import (
    initialize_feedback_system, track_agent_execution, track_tool_execution,
    FeedbackEnabledAgent, FeedbackEnabledTool, collect_rating_feedback,
    collect_binary_feedback, collect_text_feedback, collect_detailed_feedback
)


class TestResearchAgent(FeedbackEnabledAgent):
    """Test research agent with feedback collection"""
    
    def __init__(self):
        super().__init__("research_agent")
        self.search_quality = 0.8  # Initial quality parameter
        self.response_speed = 1.0   # Speed multiplier
    
    def search_web(self, query: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Search the web with feedback tracking"""
        return self.execute_with_feedback(
            operation="web_search",
            func=self._do_web_search,
            user_id=user_id,
            session_id=session_id,
            input_data={"query": query},
            query=query
        )
    
    def analyze_content(self, content: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Analyze content with feedback tracking"""
        return self.execute_with_feedback(
            operation="content_analysis",
            func=self._do_content_analysis,
            user_id=user_id,
            session_id=session_id,
            input_data={"content_length": len(content)},
            content=content
        )
    
    def _do_web_search(self, query: str) -> Dict[str, Any]:
        # Simulate search with variable quality and speed
        search_time = random.uniform(0.5, 3.0) / self.response_speed
        time.sleep(search_time)
        
        # Simulate occasional failures based on quality
        if random.random() > self.search_quality:
            raise Exception("Search API timeout or error")
        
        # Generate results based on quality
        num_results = max(1, int(5 * self.search_quality + random.uniform(-1, 1)))
        results = [f"Result {i} for '{query}'" for i in range(num_results)]
        
        return {
            "query": query,
            "results": results,
            "search_time": search_time,
            "quality_score": self.search_quality
        }
    
    def _do_content_analysis(self, content: str) -> Dict[str, Any]:
        # Simulate analysis time based on content length and speed
        analysis_time = (len(content) * 0.001) / self.response_speed
        time.sleep(analysis_time)
        
        # Simulate analysis quality
        confidence = min(0.95, self.search_quality + random.uniform(-0.1, 0.1))
        
        return {
            "content_length": len(content),
            "analysis_time": analysis_time,
            "sentiment": random.choice(["positive", "negative", "neutral"]),
            "confidence": confidence,
            "key_topics": ["topic1", "topic2", "topic3"]
        }
    
    def adapt_from_feedback(self, adaptations: List[Dict[str, Any]]):
        """Apply learning adaptations to improve performance"""
        for adaptation in adaptations:
            if adaptation.get("component_id") == self.agent_id:
                adjustments = adaptation.get("adaptation", {}).get("adjustments", {})
                
                # Apply parameter adjustments
                if "quality_threshold_increase" in adjustments:
                    self.search_quality = min(1.0, self.search_quality + adjustments["quality_threshold_increase"])
                    logging.info(f"Increased search quality to {self.search_quality:.2f}")
                
                if "timeout_reduction" in adjustments:
                    self.response_speed *= (1 / adjustments["timeout_reduction"])
                    logging.info(f"Improved response speed to {self.response_speed:.2f}x")


class TestDataProcessor(FeedbackEnabledTool):
    """Test data processing tool with feedback collection"""
    
    def __init__(self):
        super().__init__("data_processor")
        self.processing_accuracy = 0.9
        self.batch_size = 10
    
    def clean_data(self, data: List[Dict], user_id: str = None, session_id: str = None) -> List[Dict]:
        """Clean data with feedback tracking"""
        return self.execute_with_feedback(
            operation="data_cleaning",
            func=self._do_clean_data,
            user_id=user_id,
            session_id=session_id,
            input_data={"data_count": len(data)},
            data=data
        )
    
    def transform_data(self, data: List[Dict], user_id: str = None, session_id: str = None) -> List[Dict]:
        """Transform data with feedback tracking"""
        return self.execute_with_feedback(
            operation="data_transformation",
            func=self._do_transform_data,
            user_id=user_id,
            session_id=session_id,
            input_data={"data_count": len(data)},
            data=data
        )
    
    def _do_clean_data(self, data: List[Dict]) -> List[Dict]:
        # Simulate processing time based on batch size
        processing_time = len(data) / self.batch_size * 0.1
        time.sleep(processing_time)
        
        # Simulate occasional processing errors
        if random.random() > self.processing_accuracy:
            raise Exception("Data validation failed")
        
        # Simulate cleaning (remove some records based on accuracy)
        cleaned_ratio = min(0.95, self.processing_accuracy + random.uniform(-0.05, 0.05))
        cleaned_count = int(len(data) * cleaned_ratio)
        
        return data[:cleaned_count]
    
    def _do_transform_data(self, data: List[Dict]) -> List[Dict]:
        # Simulate transformation time
        processing_time = len(data) / self.batch_size * 0.05
        time.sleep(processing_time)
        
        # Transform data (add processed flag and timestamp)
        transformed = []
        for item in data:
            transformed_item = {
                **item,
                "processed": True,
                "timestamp": time.time(),
                "accuracy": self.processing_accuracy
            }
            transformed.append(transformed_item)
        
        return transformed
    
    def adapt_from_feedback(self, adaptations: List[Dict[str, Any]]):
        """Apply learning adaptations"""
        for adaptation in adaptations:
            if adaptation.get("component_id") == self.tool_id:
                adjustments = adaptation.get("adaptation", {}).get("adjustments", {})
                
                if "batch_size_increase" in adjustments:
                    self.batch_size = int(self.batch_size * adjustments["batch_size_increase"])
                    logging.info(f"Increased batch size to {self.batch_size}")
                
                if "quality_threshold_increase" in adjustments:
                    self.processing_accuracy = min(1.0, self.processing_accuracy + adjustments["quality_threshold_increase"])
                    logging.info(f"Improved processing accuracy to {self.processing_accuracy:.2f}")


# Decorator-based examples
@track_agent_execution("nlp_agent", "extract_entities")
def extract_entities(text: str, user_id: str = None, session_id: str = None) -> List[str]:
    """Extract named entities from text"""
    processing_time = len(text) * 0.0001
    time.sleep(processing_time)
    
    # Simulate occasional failures for very long texts
    if len(text) > 1000 and random.random() < 0.1:
        raise Exception("Text too long for processing")
    
    # Return mock entities
    entities = ["Entity1", "Entity2", "Entity3"]
    return entities


@track_tool_execution("file_manager", "read_file")
def read_file(file_path: str, user_id: str = None, session_id: str = None) -> str:
    """Read file content"""
    # Simulate file reading
    file_size = random.randint(100, 10000)
    read_time = file_size * 0.00001
    time.sleep(read_time)
    
    # Simulate occasional file not found errors
    if random.random() < 0.05:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return f"Content of {file_path} ({file_size} bytes)"


def simulate_user_workflow(user_id: str, session_id: str, research_agent: TestResearchAgent, 
                          data_processor: TestDataProcessor) -> Dict[str, Any]:
    """Simulate a complete user workflow"""
    
    try:
        # Step 1: Research
        search_results = research_agent.search_web(
            query="machine learning best practices",
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 2: Content analysis
        sample_content = "This is sample content for analysis. " * 20
        analysis_results = research_agent.analyze_content(
            content=sample_content,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 3: Data processing
        mock_data = [{"id": i, "content": f"data_{i}"} for i in range(15)]
        cleaned_data = data_processor.clean_data(
            data=mock_data,
            user_id=user_id,
            session_id=session_id
        )
        
        transformed_data = data_processor.transform_data(
            data=cleaned_data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 4: Entity extraction
        sample_text = "This is a sample text for entity extraction with some named entities."
        entities = extract_entities(
            text=sample_text,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 5: File operations
        file_content = read_file(
            file_path="/path/to/sample/file.txt",
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "status": "success",
            "search_results_count": len(search_results["results"]),
            "analysis_confidence": analysis_results["confidence"],
            "processed_records": len(transformed_data),
            "extracted_entities": len(entities),
            "file_size": len(file_content)
        }
        
    except Exception as e:
        logging.error(f"Workflow failed for user {user_id}: {e}")
        return {"status": "error", "error": str(e)}


def simulate_user_feedback(feedback_system, user_id: str, num_feedback: int = 5):
    """Simulate realistic user feedback"""
    
    # Get recent tasks for the user
    recent_tasks = feedback_system.collector.get_pending_tasks_for_user(user_id, limit=20)
    
    if not recent_tasks:
        logging.warning(f"No recent tasks found for user {user_id}")
        return
    
    feedback_count = 0
    for task in recent_tasks[:num_feedback]:
        
        # Determine feedback based on task outcome and performance
        if task.outcome == TaskOutcome.SUCCESS:
            # Successful tasks get better ratings
            if task.execution_time < 1.0:
                # Fast tasks get high ratings
                rating = random.uniform(4.0, 5.0)
                satisfaction = True
                text = random.choice([
                    "Very fast and accurate!",
                    "Excellent results",
                    "Perfect, exactly what I needed",
                    "Great job!"
                ])
            elif task.execution_time < 3.0:
                # Medium speed tasks get good ratings
                rating = random.uniform(3.5, 4.5)
                satisfaction = True
                text = random.choice([
                    "Good results",
                    "Helpful output",
                    "Satisfied with the outcome",
                    "Works well"
                ])
            else:
                # Slow tasks get lower ratings
                rating = random.uniform(2.5, 3.5)
                satisfaction = random.choice([True, False])
                text = random.choice([
                    "Results are good but too slow",
                    "Takes too long to complete",
                    "Could be faster",
                    "Accurate but slow"
                ])
        else:
            # Failed tasks get poor ratings
            rating = random.uniform(1.0, 2.5)
            satisfaction = False
            text = random.choice([
                "Didn't work as expected",
                "Failed to complete the task",
                "Error occurred",
                "Needs improvement"
            ])
        
        # Choose feedback type randomly
        feedback_type = random.choice([
            FeedbackType.RATING,
            FeedbackType.BINARY,
            FeedbackType.TEXT,
            FeedbackType.DETAILED
        ])
        
        try:
            if feedback_type == FeedbackType.RATING:
                collect_rating_feedback(
                    task_execution_id=task.id,
                    user_id=user_id,
                    rating=rating,
                    text_feedback=text,
                    tags=["test", "simulation"]
                )
            
            elif feedback_type == FeedbackType.BINARY:
                collect_binary_feedback(
                    task_execution_id=task.id,
                    user_id=user_id,
                    satisfied=satisfaction,
                    text_feedback=text,
                    tags=["test", "simulation"]
                )
            
            elif feedback_type == FeedbackType.TEXT:
                collect_text_feedback(
                    task_execution_id=task.id,
                    user_id=user_id,
                    text_feedback=text,
                    tags=["test", "simulation"]
                )
            
            elif feedback_type == FeedbackType.DETAILED:
                detailed_scores = {
                    "accuracy": rating,
                    "speed": max(1.0, 6.0 - task.execution_time),
                    "usefulness": rating + random.uniform(-0.5, 0.5)
                }
                collect_detailed_feedback(
                    task_execution_id=task.id,
                    user_id=user_id,
                    detailed_scores=detailed_scores,
                    text_feedback=text,
                    tags=["test", "simulation"]
                )
            
            feedback_count += 1
            
        except Exception as e:
            logging.error(f"Failed to collect feedback: {e}")
    
    logging.info(f"Collected {feedback_count} feedback items for user {user_id}")


def run_feedback_simulation(duration_minutes: int = 5, num_users: int = 3):
    """Run a comprehensive feedback simulation"""
    
    print(f"🔄 Running feedback simulation for {duration_minutes} minutes with {num_users} users")
    
    # Initialize feedback system
    feedback_system = initialize_feedback_system("test_feedback_simulation.db")
    
    # Create test components
    research_agent = TestResearchAgent()
    data_processor = TestDataProcessor()
    
    # Simulation state
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    workflow_count = 0
    
    def user_worker(user_id: str):
        nonlocal workflow_count
        session_id = f"session_{user_id}_{int(time.time())}"
        
        while time.time() < end_time:
            try:
                # Execute workflow
                result = simulate_user_workflow(user_id, session_id, research_agent, data_processor)
                workflow_count += 1
                
                if workflow_count % 10 == 0:
                    print(f"  Completed {workflow_count} workflows...")
                
                # Simulate user providing feedback (30% chance)
                if random.random() < 0.3:
                    simulate_user_feedback(feedback_system, user_id, num_feedback=2)
                
                # Wait before next workflow
                time.sleep(random.uniform(2, 8))
                
            except Exception as e:
                logging.error(f"User workflow error: {e}")
    
    # Start user threads
    threads = []
    for i in range(num_users):
        user_id = f"user_{i+1}"
        thread = threading.Thread(target=user_worker, args=(user_id,), daemon=True)
        thread.start()
        threads.append(thread)
    
    # Wait for simulation to complete
    for thread in threads:
        thread.join()
    
    print(f"✅ Simulation completed. Total workflows: {workflow_count}")
    
    return feedback_system, research_agent, data_processor


def analyze_feedback_results(feedback_system, research_agent: TestResearchAgent, 
                           data_processor: TestDataProcessor):
    """Analyze feedback results and demonstrate learning"""
    
    print("\n📊 Analyzing Feedback Results")
    print("=" * 50)
    
    # Get overall feedback statistics
    feedback_list = feedback_system.collector.storage.get_feedback(limit=1000)
    executions = feedback_system.collector.storage.get_task_executions(limit=1000)
    
    print(f"Total Executions: {len(executions)}")
    print(f"Total Feedback: {len(feedback_list)}")
    print(f"Feedback Rate: {len(feedback_list)/len(executions)*100:.1f}%")
    
    # Analyze component performance
    print(f"\n📈 Component Performance Analysis:")
    
    components = ["research_agent", "data_processor", "nlp_agent", "file_manager"]
    for component_id in components:
        profile = feedback_system.analyzer.analyze_component_performance(component_id)
        if profile.total_executions > 0:
            print(f"\n{component_id}:")
            print(f"  Executions: {profile.total_executions}")
            print(f"  Feedback Count: {profile.total_feedback_count}")
            print(f"  Average Rating: {profile.average_rating:.2f}")
            print(f"  Satisfaction Rate: {profile.satisfaction_rate:.1f}%")
            print(f"  Improvement Trend: {profile.improvement_trend:.1f}%")
            
            if profile.strengths:
                print(f"  Strengths: {', '.join(profile.strengths)}")
            if profile.weaknesses:
                print(f"  Weaknesses: {', '.join(profile.weaknesses)}")
            if profile.recommendations:
                print(f"  Recommendations: {', '.join(profile.recommendations)}")
    
    # Generate learning insights
    print(f"\n🧠 Learning Insights:")
    insights = feedback_system.analyzer.generate_learning_insights()
    
    if insights:
        for insight in insights:
            print(f"\n- {insight.component_id} ({insight.insight_type}):")
            print(f"  Description: {insight.description}")
            print(f"  Confidence: {insight.confidence:.2f}")
            print(f"  Impact Score: {insight.impact_score:.1f}/10")
            print(f"  Recommendations: {', '.join(insight.recommendations)}")
    else:
        print("No significant insights generated yet. More feedback needed.")
    
    # Apply learning strategies
    print(f"\n🎯 Applying Learning Strategies:")
    adaptations = feedback_system.learning_engine.apply_learning()
    
    if adaptations:
        print(f"Applied {len(adaptations)} learning adaptations:")
        for adaptation in adaptations:
            print(f"- {adaptation['strategy']}: {adaptation['adaptation']['type']}")
        
        # Apply adaptations to components
        research_agent.adapt_from_feedback(adaptations)
        data_processor.adapt_from_feedback(adaptations)
        
    else:
        print("No learning adaptations applied yet.")
    
    # Show adaptation history
    history = feedback_system.learning_engine.get_adaptation_history()
    if history:
        print(f"\n📚 Learning History ({len(history)} adaptations):")
        for adaptation in history[-5:]:  # Show last 5
            print(f"- {adaptation['strategy']}: {adaptation['adaptation']['type']}")


def demonstrate_manual_feedback():
    """Demonstrate manual feedback collection"""
    
    print("\n📝 Manual Feedback Collection Demo")
    print("=" * 40)
    
    feedback_system = initialize_feedback_system("manual_feedback_demo.db")
    
    # Create a sample task execution
    from feedback_system import TaskExecution
    import uuid
    
    execution = TaskExecution(
        id=str(uuid.uuid4()),
        component_id="demo_agent",
        component_type=ComponentType.AGENT,
        operation="demo_task",
        input_data={"query": "demo query"},
        output_data={"result": "demo result"},
        outcome=TaskOutcome.SUCCESS,
        execution_time=1.5,
        timestamp=time.time(),
        user_id="demo_user"
    )
    
    feedback_system.collector.register_task_execution(execution)
    
    # Collect different types of feedback
    print("Collecting various types of feedback...")
    
    # Rating feedback
    rating_feedback = collect_rating_feedback(
        task_execution_id=execution.id,
        user_id="demo_user",
        rating=4.5,
        text_feedback="Great results, very helpful!",
        tags=["helpful", "accurate"]
    )
    print(f"✅ Rating feedback collected: {rating_feedback.id}")
    
    # Binary feedback
    binary_feedback = collect_binary_feedback(
        task_execution_id=execution.id,
        user_id="demo_user",
        satisfied=True,
        text_feedback="Thumbs up!",
        tags=["satisfied"]
    )
    print(f"✅ Binary feedback collected: {binary_feedback.id}")
    
    # Detailed feedback
    detailed_feedback = collect_detailed_feedback(
        task_execution_id=execution.id,
        user_id="demo_user",
        detailed_scores={
            "accuracy": 4.5,
            "speed": 4.0,
            "usefulness": 5.0
        },
        text_feedback="Excellent accuracy and very useful results. Could be slightly faster.",
        tags=["detailed", "comprehensive"]
    )
    print(f"✅ Detailed feedback collected: {detailed_feedback.id}")
    
    print("Manual feedback collection completed!")


def main():
    """Main demonstration function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 AgentKen User Feedback Loop System Test")
    print("=" * 60)
    
    # Demonstrate manual feedback collection
    demonstrate_manual_feedback()
    
    # Run comprehensive simulation
    print(f"\n🔄 Starting comprehensive feedback simulation...")
    feedback_system, research_agent, data_processor = run_feedback_simulation(
        duration_minutes=2,  # Short duration for demo
        num_users=2
    )
    
    # Wait for final feedback collection
    print("\n⏳ Processing final feedback...")
    time.sleep(3)
    
    # Analyze results
    analyze_feedback_results(feedback_system, research_agent, data_processor)
    
    print(f"\n🎉 Feedback system demonstration complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"✅ Automatic task execution tracking")
    print(f"✅ Multiple feedback types (rating, binary, text, detailed)")
    print(f"✅ Performance analysis and profiling")
    print(f"✅ Learning insight generation")
    print(f"✅ Continuous learning and adaptation")
    print(f"✅ Component improvement based on feedback")
    
    print(f"\nFeedback data stored in: test_feedback_simulation.db")
    print(f"To view the web interface, run:")
    print(f"  python core/feedback_interface.py")


if __name__ == "__main__":
    main()