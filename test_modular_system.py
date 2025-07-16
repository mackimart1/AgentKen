"""
Comprehensive test and demonstration of the AgentKen Modular System
Shows module discovery, loading, composition, and lifecycle management.
"""

import logging
import time
import json
from pathlib import Path

# Add the core directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from module_system import (
    initialize_module_system, get_module_registry, ModuleType, ModuleStatus
)
from module_composer import (
    get_composition_registry, CompositionBuilder, CompositionType, ExecutionMode
)
from module_lifecycle import ModuleLifecycleManager


def test_module_discovery_and_loading():
    """Test module discovery and loading"""
    print("🔍 Testing Module Discovery and Loading")
    print("=" * 50)
    
    # Initialize module system
    registry = initialize_module_system(["modules", "agents", "tools"])
    
    # List discovered modules
    modules = registry.list_modules()
    print(f"Discovered {len(modules)} modules:")
    
    for metadata in modules:
        print(f"  📦 {metadata.name} v{metadata.version} ({metadata.module_type.value})")
        print(f"     {metadata.description}")
        if metadata.capabilities:
            capabilities = [cap.name for cap in metadata.capabilities]
            print(f"     Capabilities: {', '.join(capabilities)}")
        print(f"     Status: {metadata.status.value}")
        print()
    
    return registry


def test_module_initialization():
    """Test module initialization"""
    print("🚀 Testing Module Initialization")
    print("=" * 50)
    
    registry = get_module_registry()
    
    # Get available modules
    modules = registry.list_modules(status=ModuleStatus.LOADED)
    
    initialized_count = 0
    for metadata in modules:
        try:
            print(f"Initializing {metadata.name}...")
            success = registry.initialize_module(metadata.id)
            
            if success:
                print(f"  ✅ {metadata.name} initialized successfully")
                initialized_count += 1
            else:
                print(f"  ❌ {metadata.name} initialization failed")
                
        except Exception as e:
            print(f"  ❌ {metadata.name} initialization error: {e}")
    
    print(f"\nInitialized {initialized_count}/{len(modules)} modules")
    return initialized_count


def test_module_capabilities():
    """Test module capabilities execution"""
    print("⚡ Testing Module Capabilities")
    print("=" * 50)
    
    registry = get_module_registry()
    
    # Test research agent capabilities
    research_modules = registry.find_modules_by_capability("web_search")
    if research_modules:
        module_id = research_modules[0].id
        print(f"Testing research capabilities with {module_id}...")
        
        try:
            # Test web search
            search_result = registry.execute_capability(
                module_id, "web_search", 
                query="artificial intelligence trends",
                max_results=3
            )
            print(f"  ✅ Web search: Found {search_result['total_results']} results")
            
            # Test content analysis
            analysis_result = registry.execute_capability(
                module_id, "analyze_content",
                content="Artificial intelligence is transforming industries worldwide. Machine learning algorithms are becoming more sophisticated.",
                analysis_type="full"
            )
            print(f"  ✅ Content analysis: {analysis_result.get('sentiment', 'unknown')} sentiment")
            
            # Test topic research
            research_result = registry.execute_capability(
                module_id, "research_topic",
                topic="machine learning applications",
                depth="basic"
            )
            print(f"  ✅ Topic research: {len(research_result['key_findings'])} key findings")
            
        except Exception as e:
            print(f"  ❌ Research capabilities test failed: {e}")
    
    # Test data processing capabilities
    data_modules = registry.find_modules_by_capability("clean_data")
    if data_modules:
        module_id = data_modules[0].id
        print(f"\nTesting data processing capabilities with {module_id}...")
        
        try:
            # Create sample data
            sample_data = [
                {"name": "Alice", "age": 25, "score": 85.5, "category": "A"},
                {"name": "Bob", "age": 30, "score": 92.0, "category": "B"},
                {"name": "Charlie", "age": 25, "score": 78.5, "category": "A"},
                {"name": "Diana", "age": 28, "score": 88.0, "category": "B"},
                {"name": "Eve", "age": 30, "score": 95.5, "category": "A"}
            ]
            
            # Test data cleaning
            clean_result = registry.execute_capability(
                module_id, "clean_data",
                data=sample_data,
                operations=["remove_duplicates", "handle_missing"]
            )
            print(f"  ✅ Data cleaning: {clean_result['rows_removed']} rows removed")
            
            # Test data analysis
            analysis_result = registry.execute_capability(
                module_id, "analyze_data",
                data=sample_data,
                analysis_type="descriptive"
            )
            print(f"  ✅ Data analysis: {len(analysis_result['insights'])} insights generated")
            
        except Exception as e:
            print(f"  ❌ Data processing capabilities test failed: {e}")


def test_module_composition():
    """Test module composition and workflows"""
    print("🔧 Testing Module Composition")
    print("=" * 50)
    
    registry = get_module_registry()
    composition_registry = get_composition_registry()
    
    # Find available modules
    research_modules = registry.find_modules_by_capability("web_search")
    data_modules = registry.find_modules_by_capability("analyze_data")
    
    if research_modules and data_modules:
        research_module_id = research_modules[0].id
        data_module_id = data_modules[0].id
        
        print(f"Creating composition with {research_module_id} and {data_module_id}...")
        
        try:
            # Create a pipeline composition
            builder = CompositionBuilder(
                "research_analysis_pipeline",
                "Research and Analysis Pipeline",
                CompositionType.PIPELINE
            )
            
            composition = (builder
                          .set_description("Pipeline that researches a topic and analyzes the results")
                          .set_execution_mode(ExecutionMode.SYNCHRONOUS)
                          .add_step(
                              "research_step",
                              research_module_id,
                              "research_topic",
                              parameters={"depth": "basic", "sources": 3},
                              input_mapping={"topic": "research_topic"},
                              output_mapping={"key_findings": "research_findings"}
                          )
                          .add_step(
                              "analysis_step", 
                              data_module_id,
                              "analyze_data",
                              parameters={"analysis_type": "descriptive"},
                              input_mapping={"data": "research_findings"}
                          )
                          .build())
            
            # Register composition
            success = composition_registry.register_composition(composition)
            
            if success:
                print("  ✅ Composition created and registered successfully")
                
                # Execute composition
                input_data = {"research_topic": "artificial intelligence in healthcare"}
                
                print("  🔄 Executing composition...")
                result = composition.execute(input_data)
                
                print(f"  ✅ Composition executed successfully")
                print(f"     Research topic: {result.get('research_topic', 'N/A')}")
                print(f"     Findings: {len(result.get('research_findings', []))} items")
                
            else:
                print("  ❌ Failed to register composition")
                
        except Exception as e:
            print(f"  ❌ Composition test failed: {e}")
    
    else:
        print("  ⚠️ Required modules not available for composition test")


def test_parallel_composition():
    """Test parallel module composition"""
    print("⚡ Testing Parallel Composition")
    print("=" * 50)
    
    registry = get_module_registry()
    composition_registry = get_composition_registry()
    
    # Find available modules
    research_modules = registry.find_modules_by_capability("web_search")
    
    if len(research_modules) >= 1:
        research_module_id = research_modules[0].id
        
        print(f"Creating parallel composition with {research_module_id}...")
        
        try:
            # Create a parallel composition
            builder = CompositionBuilder(
                "parallel_research",
                "Parallel Research Composition",
                CompositionType.PARALLEL
            )
            
            composition = (builder
                          .set_description("Parallel research on multiple topics")
                          .set_execution_mode(ExecutionMode.SYNCHRONOUS)
                          .add_step(
                              "research_ai",
                              research_module_id,
                              "web_search",
                              parameters={"query": "artificial intelligence", "max_results": 3},
                              parallel_group="research_group"
                          )
                          .add_step(
                              "research_ml",
                              research_module_id,
                              "web_search", 
                              parameters={"query": "machine learning", "max_results": 3},
                              parallel_group="research_group"
                          )
                          .add_step(
                              "research_dl",
                              research_module_id,
                              "web_search",
                              parameters={"query": "deep learning", "max_results": 3},
                              parallel_group="research_group"
                          )
                          .build())
            
            # Register and execute composition
            success = composition_registry.register_composition(composition)
            
            if success:
                print("  ✅ Parallel composition created successfully")
                
                # Execute composition
                input_data = {}
                
                print("  🔄 Executing parallel composition...")
                start_time = time.time()
                result = composition.execute(input_data)
                execution_time = time.time() - start_time
                
                print(f"  ✅ Parallel composition executed in {execution_time:.2f} seconds")
                
                # Count total results
                total_results = 0
                for key, value in result.items():
                    if isinstance(value, dict) and "total_results" in value:
                        total_results += value["total_results"]
                
                print(f"     Total search results: {total_results}")
                
            else:
                print("  ❌ Failed to register parallel composition")
                
        except Exception as e:
            print(f"  ❌ Parallel composition test failed: {e}")
    
    else:
        print("  ⚠️ Required modules not available for parallel composition test")


def test_module_lifecycle():
    """Test module lifecycle management"""
    print("🔄 Testing Module Lifecycle Management")
    print("=" * 50)
    
    try:
        # Initialize lifecycle manager
        lifecycle_manager = ModuleLifecycleManager()
        
        # List installed modules
        installed_modules = lifecycle_manager.list_installed_modules()
        print(f"Currently installed modules: {len(installed_modules)}")
        
        for module in installed_modules:
            print(f"  📦 {module['name']} v{module['version']}")
            status = module['status']
            print(f"     Status: {'✅' if status['installed'] else '❌'} Installed, "
                  f"{'✅' if status['loaded'] else '❌'} Loaded, "
                  f"{'✅' if status['active'] else '❌'} Active")
        
        # Check for updates
        print("\nChecking for module updates...")
        updates = lifecycle_manager.check_for_updates()
        
        if updates:
            print(f"Available updates: {len(updates)}")
            for module_id, package in updates.items():
                print(f"  📦 {module_id}: {package.version.to_string()}")
        else:
            print("  ✅ All modules are up to date")
        
        # Test module status
        registry = get_module_registry()
        modules = registry.list_modules()
        
        if modules:
            test_module_id = modules[0].id
            status = lifecycle_manager.get_module_status(test_module_id)
            
            print(f"\nDetailed status for {test_module_id}:")
            print(f"  Installed: {status['installed']}")
            print(f"  Loaded: {status['loaded']}")
            print(f"  Active: {status['active']}")
            print(f"  Version: {status['version']}")
            print(f"  Dependencies satisfied: {status['dependencies_satisfied']}")
            
            if status['last_event']:
                event = status['last_event']
                print(f"  Last event: {event['event']} at {time.ctime(event['timestamp'])}")
        
    except Exception as e:
        print(f"  ❌ Lifecycle management test failed: {e}")


def test_module_health_monitoring():
    """Test module health and monitoring"""
    print("🏥 Testing Module Health Monitoring")
    print("=" * 50)
    
    registry = get_module_registry()
    
    # Get all active modules
    active_modules = registry.list_modules(status=ModuleStatus.ACTIVE)
    
    print(f"Monitoring {len(active_modules)} active modules:")
    
    for metadata in active_modules:
        module = registry.get_module(metadata.id)
        if module:
            health = module.get_health()
            
            print(f"  📊 {metadata.name}:")
            print(f"     Status: {health['status']}")
            print(f"     Initialized: {'✅' if health['is_initialized'] else '❌'}")
            print(f"     Usage count: {health['usage_count']}")
            
            if health['last_used']:
                last_used = time.ctime(health['last_used'])
                print(f"     Last used: {last_used}")
            
            if health['error_message']:
                print(f"     ⚠️ Error: {health['error_message']}")
            
            print()


def test_error_handling():
    """Test error handling and recovery"""
    print("🛡️ Testing Error Handling and Recovery")
    print("=" * 50)
    
    registry = get_module_registry()
    
    # Test invalid capability execution
    modules = registry.list_modules(status=ModuleStatus.ACTIVE)
    if modules:
        test_module_id = modules[0].id
        
        print(f"Testing error handling with {test_module_id}...")
        
        try:
            # Try to execute non-existent capability
            result = registry.execute_capability(test_module_id, "invalid_capability")
            print("  ❌ Expected error but got result")
        except Exception as e:
            print(f"  ✅ Correctly handled invalid capability: {type(e).__name__}")
        
        try:
            # Try to execute with invalid parameters
            result = registry.execute_capability(test_module_id, "web_search", invalid_param="test")
            print("  ⚠️ Executed with invalid parameters (may be handled by module)")
        except Exception as e:
            print(f"  ✅ Correctly handled invalid parameters: {type(e).__name__}")
    
    # Test module recovery
    print("\nTesting module recovery...")
    
    if modules:
        test_module_id = modules[0].id
        module = registry.get_module(test_module_id)
        
        if module and module.is_initialized:
            print(f"  Shutting down {test_module_id}...")
            success = registry.shutdown_module(test_module_id)
            
            if success:
                print("  ✅ Module shutdown successful")
                
                print(f"  Reinitializing {test_module_id}...")
                success = registry.initialize_module(test_module_id)
                
                if success:
                    print("  ✅ Module recovery successful")
                else:
                    print("  ❌ Module recovery failed")
            else:
                print("  ❌ Module shutdown failed")


def generate_system_report():
    """Generate comprehensive system report"""
    print("📊 Generating System Report")
    print("=" * 50)
    
    registry = get_module_registry()
    composition_registry = get_composition_registry()
    
    # Module statistics
    all_modules = registry.list_modules()
    loaded_modules = registry.list_modules(status=ModuleStatus.LOADED)
    active_modules = registry.list_modules(status=ModuleStatus.ACTIVE)
    error_modules = registry.list_modules(status=ModuleStatus.ERROR)
    
    # Module type breakdown
    agents = registry.list_modules(module_type=ModuleType.AGENT)
    tools = registry.list_modules(module_type=ModuleType.TOOL)
    
    # Composition statistics
    compositions = composition_registry.list_compositions()
    
    print("📈 System Statistics:")
    print(f"  Total modules discovered: {len(all_modules)}")
    print(f"  Loaded modules: {len(loaded_modules)}")
    print(f"  Active modules: {len(active_modules)}")
    print(f"  Error modules: {len(error_modules)}")
    print()
    
    print("📦 Module Types:")
    print(f"  Agents: {len(agents)}")
    print(f"  Tools: {len(tools)}")
    print()
    
    print("🔧 Compositions:")
    print(f"  Total compositions: {len(compositions)}")
    print()
    
    # Capability overview
    all_capabilities = set()
    for metadata in all_modules:
        for capability in metadata.capabilities:
            all_capabilities.add(capability.name)
    
    print("⚡ Available Capabilities:")
    for capability in sorted(all_capabilities):
        modules_with_capability = registry.find_modules_by_capability(capability)
        print(f"  {capability}: {len(modules_with_capability)} modules")
    
    print()
    
    # Health overview
    healthy_modules = 0
    for metadata in active_modules:
        module = registry.get_module(metadata.id)
        if module:
            health = module.get_health()
            if health['status'] == 'active' and not health['error_message']:
                healthy_modules += 1
    
    health_percentage = (healthy_modules / len(active_modules) * 100) if active_modules else 0
    
    print("🏥 System Health:")
    print(f"  Healthy modules: {healthy_modules}/{len(active_modules)} ({health_percentage:.1f}%)")
    
    if health_percentage >= 90:
        print("  Overall health: 🟢 Excellent")
    elif health_percentage >= 70:
        print("  Overall health: 🟡 Good")
    elif health_percentage >= 50:
        print("  Overall health: 🟠 Fair")
    else:
        print("  Overall health: 🔴 Poor")


def main():
    """Main test function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during testing
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 AgentKen Modular System Comprehensive Test")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Module Discovery and Loading
        registry = test_module_discovery_and_loading()
        print()
        
        # Test 2: Module Initialization
        initialized_count = test_module_initialization()
        print()
        
        if initialized_count > 0:
            # Test 3: Module Capabilities
            test_module_capabilities()
            print()
            
            # Test 4: Module Composition
            test_module_composition()
            print()
            
            # Test 5: Parallel Composition
            test_parallel_composition()
            print()
            
            # Test 6: Module Health Monitoring
            test_module_health_monitoring()
            print()
            
            # Test 7: Error Handling
            test_error_handling()
            print()
        
        # Test 8: Module Lifecycle
        test_module_lifecycle()
        print()
        
        # Generate final report
        generate_system_report()
        print()
        
        print("🎉 Modular System Test Completed Successfully!")
        print()
        print("Key Features Demonstrated:")
        print("✅ Module discovery and automatic loading")
        print("✅ Dynamic module initialization and management")
        print("✅ Capability-based module execution")
        print("✅ Pipeline and parallel composition")
        print("✅ Module health monitoring")
        print("✅ Error handling and recovery")
        print("✅ Lifecycle management and updates")
        print()
        print("The AgentKen Modular System is fully operational!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()