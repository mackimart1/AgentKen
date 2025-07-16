"""
Test script for the AgentKen Performance Monitoring System
Demonstrates the monitoring capabilities with simulated agents and tools.
"""

import time
import random
import logging
import threading
from typing import Dict, List, Any

# Add the core directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Import the performance monitoring components
from performance_decorators import (
    initialize_performance_monitoring,
    monitor_agent_execution,
    monitor_tool_execution,
    MonitoredAgent,
    MonitoredTool,
    get_performance_monitor
)
from performance_monitor import MetricType, ComponentType


class TestResearchAgent(MonitoredAgent):
    """Test research agent with performance monitoring"""
    
    def __init__(self):
        super().__init__("research_agent")
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web for information"""
        return self.execute_with_monitoring("search_web", self._do_web_search, query)
    
    def analyze_results(self, results: List[str]) -> Dict[str, Any]:
        """Analyze search results"""
        return self.execute_with_monitoring("analyze_results", self._do_analysis, results)
    
    def _do_web_search(self, query: str) -> Dict[str, Any]:
        # Simulate web search with variable latency
        search_time = random.uniform(0.1, 0.8)
        time.sleep(search_time)
        
        # Simulate occasional failures
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Web search API timeout")
        
        # Record custom metrics
        self.record_custom_metric(
            MetricType.THROUGHPUT,
            1.0 / search_time,
            "searches_per_second",
            tags={"query_type": "web_search"}
        )
        
        return {
            "query": query,
            "results": [f"Result {i} for {query}" for i in range(5)],
            "search_time": search_time
        }
    
    def _do_analysis(self, results: List[str]) -> Dict[str, Any]:
        # Simulate analysis with processing time proportional to result count
        analysis_time = len(results) * 0.02
        time.sleep(analysis_time)
        
        # Record custom metrics
        self.record_custom_metric(
            MetricType.RESOURCE_USAGE,
            len(results) * 5,  # Simulate memory usage
            "MB",
            tags={"operation": "analysis"}
        )
        
        return {
            "analyzed_results": len(results),
            "sentiment_score": random.uniform(0.3, 0.9),
            "confidence": random.uniform(0.7, 0.95)
        }


class TestDataProcessor(MonitoredTool):
    """Test data processing tool with performance monitoring"""
    
    def __init__(self):
        super().__init__("data_processor")
    
    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """Clean and validate data"""
        return self.execute_with_monitoring("clean_data", self._do_clean, data)
    
    def transform_data(self, data: List[Dict]) -> List[Dict]:
        """Transform data format"""
        return self.execute_with_monitoring("transform_data", self._do_transform, data)
    
    def _do_clean(self, data: List[Dict]) -> List[Dict]:
        # Simulate data cleaning
        processing_time = len(data) * 0.01
        time.sleep(processing_time)
        
        # Simulate occasional processing errors
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Data validation failed")
        
        # Record throughput metric
        self.record_custom_metric(
            MetricType.THROUGHPUT,
            len(data) / processing_time,
            "records_per_second",
            tags={"operation": "cleaning"}
        )
        
        # Simulate removing some invalid records
        cleaned_count = int(len(data) * random.uniform(0.85, 0.95))
        return data[:cleaned_count]
    
    def _do_transform(self, data: List[Dict]) -> List[Dict]:
        # Simulate data transformation
        processing_time = len(data) * 0.005
        time.sleep(processing_time)
        
        # Record CPU usage metric
        cpu_usage = random.uniform(20, 80)
        self.record_custom_metric(
            MetricType.CPU_USAGE,
            cpu_usage,
            "percent",
            tags={"operation": "transformation"}
        )
        
        # Transform data (add processed flag)
        transformed = []
        for item in data:
            transformed_item = {**item, "processed": True, "timestamp": time.time()}
            transformed.append(transformed_item)
        
        return transformed


# Decorator-based examples
@monitor_agent_execution("nlp_agent", "extract_entities")
def extract_entities(text: str) -> List[str]:
    """Extract named entities from text"""
    # Simulate NLP processing
    processing_time = len(text) * 0.0001
    time.sleep(processing_time)
    
    # Simulate occasional failures for very long texts
    if len(text) > 1000 and random.random() < 0.15:
        raise Exception("Text too long for processing")
    
    # Return mock entities
    entities = ["Entity1", "Entity2", "Entity3"]
    return entities


@monitor_tool_execution("file_processor", "read_file")
def read_file(file_path: str) -> str:
    """Read file content"""
    # Simulate file reading with variable time based on "file size"
    file_size = random.randint(100, 10000)  # Simulate file size
    read_time = file_size * 0.00001
    time.sleep(read_time)
    
    # Simulate occasional file not found errors
    if random.random() < 0.08:  # 8% failure rate
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return f"Content of {file_path} ({file_size} bytes)"


def simulate_workflow():
    """Simulate a complete workflow with multiple agents and tools"""
    
    # Initialize components
    research_agent = TestResearchAgent()
    data_processor = TestDataProcessor()
    
    try:
        # Step 1: Research
        search_results = research_agent.search_web("AI performance monitoring")
        
        # Step 2: Analyze results
        analysis = research_agent.analyze_results(search_results["results"])
        
        # Step 3: Process data
        mock_data = [{"id": i, "content": f"data_{i}"} for i in range(10)]
        cleaned_data = data_processor.clean_data(mock_data)
        transformed_data = data_processor.transform_data(cleaned_data)
        
        # Step 4: Extract entities
        sample_text = "This is a sample text for entity extraction with some named entities."
        entities = extract_entities(sample_text)
        
        # Step 5: File operations
        file_content = read_file("/path/to/sample/file.txt")
        
        return {
            "search_results": len(search_results["results"]),
            "analysis_confidence": analysis["confidence"],
            "processed_records": len(transformed_data),
            "extracted_entities": len(entities),
            "file_size": len(file_content)
        }
        
    except Exception as e:
        logging.error(f"Workflow failed: {e}")
        return {"error": str(e)}


def run_load_test(duration_seconds: int = 60, concurrent_workflows: int = 3):
    """Run a load test to generate performance data"""
    
    print(f"🔄 Running load test for {duration_seconds} seconds with {concurrent_workflows} concurrent workflows")
    
    start_time = time.time()
    workflow_count = 0
    
    def worker():
        nonlocal workflow_count
        while time.time() - start_time < duration_seconds:
            try:
                result = simulate_workflow()
                workflow_count += 1
                
                if workflow_count % 10 == 0:
                    print(f"  Completed {workflow_count} workflows...")
                
                # Small delay between workflows
                time.sleep(random.uniform(0.1, 0.5))
                
            except Exception as e:
                logging.error(f"Workflow error: {e}")
    
    # Start worker threads
    threads = []
    for i in range(concurrent_workflows):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    print(f"✅ Load test completed. Total workflows: {workflow_count}")
    return workflow_count


def generate_performance_report():
    """Generate and display a performance report"""
    
    monitor = get_performance_monitor()
    if not monitor:
        print("❌ Performance monitor not initialized")
        return
    
    print("\n📊 Performance Report")
    print("=" * 50)
    
    # System overview
    overview = monitor.dashboard.generate_system_overview(time_window_hours=1)
    
    print(f"System Health Score: {monitor.get_system_health_score():.1f}/100")
    print(f"Total Executions: {overview['system_metrics']['total_executions']}")
    print(f"Success Rate: {overview['system_metrics']['system_success_rate']:.1f}%")
    print(f"Average Latency: {overview['system_metrics']['avg_system_latency']:.1f}ms")
    print(f"Active Alerts: {overview['system_metrics']['active_alerts']}")
    
    # Component performance
    print(f"\n📈 Component Performance:")
    for component_id, stats in overview['component_stats'].items():
        print(f"  {component_id}:")
        print(f"    Executions: {stats['total_executions']}")
        print(f"    Success Rate: {stats['success_rate']:.1f}%")
        print(f"    Avg Latency: {stats['avg_latency']:.1f}ms")
    
    # Top performers
    if overview['top_performers']:
        print(f"\n🏆 Top Performers:")
        for performer in overview['top_performers'][:3]:
            print(f"  {performer['component_id']} (Score: {performer['score']:.1f})")
    
    # Bottlenecks
    if overview['bottlenecks']:
        print(f"\n⚠️  Bottlenecks Detected:")
        for bottleneck in overview['bottlenecks']:
            print(f"  {bottleneck['component_id']}: {', '.join(bottleneck['issues'])}")
    
    # Active alerts
    active_alerts = monitor.storage.get_active_alerts()
    if active_alerts:
        print(f"\n🚨 Active Alerts:")
        for alert in active_alerts:
            print(f"  {alert.level.value.upper()}: {alert.title}")
            print(f"    {alert.description}")
    else:
        print(f"\n✅ No active alerts")


def main():
    """Main test function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 AgentKen Performance Monitoring Test")
    print("=" * 50)
    
    # Initialize performance monitoring
    print("📊 Initializing performance monitoring...")
    monitor = initialize_performance_monitoring("test_performance_metrics.db")
    
    if not monitor:
        print("❌ Failed to initialize performance monitoring")
        return
    
    print("✅ Performance monitoring initialized")
    
    # Run a few individual workflows first
    print("\n🔧 Running sample workflows...")
    for i in range(5):
        result = simulate_workflow()
        print(f"  Workflow {i+1}: {result}")
        time.sleep(0.5)
    
    # Run load test
    print("\n🔄 Starting load test...")
    workflow_count = run_load_test(duration_seconds=30, concurrent_workflows=2)
    
    # Wait for metrics to be processed
    print("\n⏳ Processing metrics...")
    time.sleep(3)
    
    # Generate performance report
    generate_performance_report()
    
    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Dashboard available at: http://localhost:5001")
    print(f"💾 Metrics stored in: test_performance_metrics.db")
    print(f"\nTo view the dashboard, run:")
    print(f"  python setup_performance_monitoring.py --dashboard")
    
    # Stop monitoring
    monitor.stop()


if __name__ == "__main__":
    main()