#!/usr/bin/env python3
"""
Comprehensive test for the full Inferra V Enhanced System with legacy integration
"""

import logging
import time
import json
from typing import Dict, Any

def test_legacy_integration():
    """Test legacy component integration"""
    print("🔗 Testing Legacy Integration")
    print("=" * 50)
    
    try:
        from core.legacy_integration import LegacyIntegrationManager
        
        manager = LegacyIntegrationManager()
        manager.discover_legacy_components()
        
        report = manager.get_integration_report()
        print(f"✅ Discovered {report['discovered_agents']} agents and {report['discovered_tools']} tools")
        
        # Show some examples
        if report['agent_names']:
            print(f"📋 Sample agents: {', '.join(report['agent_names'][:5])}")
        if report['tool_names']:
            print(f"🛠️ Sample tools: {', '.join(report['tool_names'][:5])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy integration test failed: {e}")
        return False

def test_ai_communication():
    """Test AI communication system"""
    print("\n🤖 Testing AI Communication")
    print("=" * 50)
    
    try:
        from core.ai_communication import get_ai_assistant
        
        assistant = get_ai_assistant()
        
        # Test basic chat
        response = assistant.chat("Hello, can you help me?")
        print(f"✅ AI Response: {response[:100]}...")
        
        # Test with system context
        system_state = {
            "health_score": 95.0,
            "active_plans": 2,
            "agents": 15
        }
        
        response = assistant.chat("What's the system status?", system_state=system_state)
        print(f"✅ Contextual Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ AI communication test failed: {e}")
        return False

def test_enhanced_orchestrator():
    """Test the enhanced orchestrator"""
    print("\n🎯 Testing Enhanced Orchestrator")
    print("=" * 50)
    
    try:
        from core.workflow_monitoring import WorkflowMonitor
        from core.tool_integration_system import create_tool_system
        from core.agent_framework import create_agent_system
        from core.adaptive_orchestrator import AdvancedOrchestrator, create_sample_plan
        from core.legacy_integration import integrate_legacy_components
        
        # Initialize components
        monitor = WorkflowMonitor()
        monitor.start()
        
        tool_registry = create_tool_system()
        message_bus, _, agents = create_agent_system()
        
        # Integrate legacy components
        integration_report = integrate_legacy_components(tool_registry, agents, message_bus)
        print(f"✅ Integrated {integration_report['wrapped_agents']} agents and {integration_report['wrapped_tools']} tools")
        
        # Create orchestrator
        orchestrator = AdvancedOrchestrator(message_bus, tool_registry)
        
        # Create a simple plan
        plan_data = create_sample_plan()
        plan_id = orchestrator.create_plan(**plan_data)
        print(f"✅ Created plan: {plan_id}")
        
        # Get system metrics
        metrics = orchestrator.get_system_metrics()
        print(f"✅ System metrics: {metrics['total_plans']} plans, {len(agents)} agents")
        
        # Cleanup
        monitor.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_interface_components():
    """Test user interface components"""
    print("\n💬 Testing User Interface Components")
    print("=" * 50)
    
    try:
        from user_interface import InferraVInterface
        
        # Create interface instance
        interface = InferraVInterface()
        
        # Test some basic methods
        print("✅ Interface created successfully")
        
        # Test configuration loading
        from config import config
        print(f"✅ Configuration loaded: {config.model.provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ User interface test failed: {e}")
        return False

def test_web_api_components():
    """Test web API components"""
    print("\n🌐 Testing Web API Components")
    print("=" * 50)
    
    try:
        from web_api import InferraVAPI
        
        # Create API instance
        api = InferraVAPI()
        print("✅ Web API created successfully")
        
        # Test client components
        from client_example import InferraVClient
        
        # Create client (don't connect)
        client = InferraVClient()
        print("✅ API client created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Web API test failed: {e}")
        return False

def test_monitoring_system():
    """Test monitoring and optimization system"""
    print("\n📊 Testing Monitoring System")
    print("=" * 50)
    
    try:
        from core.workflow_monitoring import WorkflowMonitor, Metric, MetricType
        
        # Create monitor
        monitor = WorkflowMonitor()
        monitor.start()
        
        # Add a test metric
        test_metric = Metric(
            name="test_metric",
            type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            unit="requests_per_second"
        )
        
        monitor.metrics_collector.record_metric(test_metric)
        
        # Get health score
        health_score = monitor.get_system_health_score()
        print(f"✅ System health score: {health_score:.1f}/100")
        
        # Get optimization report
        report = monitor.get_optimization_report()
        print(f"✅ Optimization report: {len(report['optimization_recommendations'])} recommendations")
        
        # Cleanup
        monitor.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 INFERRA V ENHANCED SYSTEM - COMPREHENSIVE TEST")
    print("=" * 80)
    
    tests = [
        ("Legacy Integration", test_legacy_integration),
        ("AI Communication", test_ai_communication),
        ("Enhanced Orchestrator", test_enhanced_orchestrator),
        ("User Interface", test_user_interface_components),
        ("Web API", test_web_api_components),
        ("Monitoring System", test_monitoring_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is fully operational.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    success = run_comprehensive_test()
    exit(0 if success else 1)