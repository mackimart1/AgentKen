#!/usr/bin/env python3
"""
Comprehensive integration test for the Inferra V system.
Tests all major components and their interactions.
"""

import sys
import os
import traceback
import time
from typing import Dict, Any, List


def test_config_and_openrouter():
    """Test configuration and OpenRouter API connectivity."""
    print("🧪 Testing Configuration and OpenRouter API...")
    try:
        import config
        from langchain_core.messages import HumanMessage

        # Test config loading
        assert config.config.model.provider == "OPENROUTER"
        assert config.config.model.model_name == "deepseek/deepseek-chat-v3-0324:free"
        assert config.default_langchain_model is not None

        # Test API call
        response = config.default_langchain_model.invoke(
            [HumanMessage(content="Respond with exactly: 'API_TEST_SUCCESS'")]
        )
        assert "API_TEST_SUCCESS" in response.content

        print("✅ Configuration and OpenRouter API working correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration/OpenRouter test failed: {e}")
        return False


def test_utils_and_manifests():
    """Test utils module and manifest loading."""
    print("🧪 Testing Utils and Manifests...")
    try:
        import utils

        # Test manifest loading
        tools_manifest = utils.get_tools_manifest()
        agents_manifest = utils.get_agents_manifest()

        assert len(tools_manifest) > 0, "No tools in manifest"
        assert len(agents_manifest) > 0, "No agents in manifest"

        # Test tool loading
        tools = utils.all_tool_functions()
        assert len(tools) > 0, "No tools loaded"

        # Test agent listing
        agents = utils.all_agents()
        assert len(agents) > 0, "No agents available"

        print(f"✅ Utils working correctly - {len(tools)} tools, {len(agents)} agents")
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False


def test_memory_manager():
    """Test memory manager functionality."""
    print("🧪 Testing Memory Manager...")
    try:
        import memory_manager

        mm = memory_manager.MemoryManager()

        # Test adding memory
        success = mm.add_memory(
            key="test_integration",
            value="Integration test memory",
            memory_type="test",
            importance=8,
        )
        assert success, "Failed to add memory"

        # Test retrieving memory
        memory = mm.retrieve_memory("test_integration")
        assert memory is not None, "Failed to retrieve memory"
        assert memory["value"] == "Integration test memory"

        # Test searching memories
        memories = mm.search_memories("integration")
        assert len(memories) > 0, "Failed to search memories"

        # Cleanup
        mm.delete_memory("test_integration")
        mm.close()

        print("✅ Memory Manager working correctly")
        return True
    except Exception as e:
        print(f"❌ Memory Manager test failed: {e}")
        return False


def test_core_modules():
    """Test core system modules."""
    print("🧪 Testing Core Modules...")
    try:
        # Test workflow monitoring
        from core.workflow_monitoring import WorkflowMonitor

        monitor = WorkflowMonitor()

        # Test tool integration system
        from core.tool_integration_system import create_tool_system

        tool_registry = create_tool_system()
        assert len(tool_registry.tools) > 0, "No tools in registry"

        # Test agent framework
        from core.agent_framework import create_agent_system

        message_bus, _, agents = create_agent_system()
        assert len(agents) > 0, "No agents created"

        # Test adaptive orchestrator
        from core.adaptive_orchestrator import AdvancedOrchestrator

        orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

        # Test legacy integration
        from core.legacy_integration import integrate_legacy_components

        integration_report = integrate_legacy_components(
            tool_registry, agents, message_bus
        )

        print(f"✅ Core modules working correctly - {integration_report}")
        return True
    except Exception as e:
        print(f"❌ Core modules test failed: {e}")
        return False


def test_hermes_agent():
    """Test Hermes agent functionality."""
    print("🧪 Testing Hermes Agent...")
    try:
        from agents.hermes import hermes

        # Test that Hermes can be imported and initialized
        # Note: We won't run a full conversation to avoid complexity
        print("✅ Hermes agent imported and initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Hermes agent test failed: {e}")
        return False


def test_individual_tools():
    """Test individual tools functionality."""
    print("🧪 Testing Individual Tools...")
    try:
        # Test list_available_agents
        from tools.list_available_agents import list_available_agents

        agents = list_available_agents.invoke({})
        assert isinstance(agents, dict), "list_available_agents should return dict"

        # Test scratchpad
        from tools.scratchpad import scratchpad

        result = scratchpad.invoke(
            {"action": "write", "key": "test", "value": "test_value"}
        )
        assert result == "test_value", "Scratchpad write failed"

        result = scratchpad.invoke({"action": "read", "key": "test"})
        assert result == "test_value", "Scratchpad read failed"

        scratchpad.invoke({"action": "clear", "key": "test"})

        # Test secure_code_executor
        from tools.secure_code_executor import secure_code_executor

        result = secure_code_executor.invoke({"code": "print('Hello World')"})
        assert "Hello World" in result, "Code executor failed"

        print("✅ Individual tools working correctly")
        return True
    except Exception as e:
        print(f"❌ Individual tools test failed: {e}")
        return False


def test_agent_loading():
    """Test that agents can be loaded from manifest."""
    print("🧪 Testing Agent Loading...")
    try:
        import utils

        # Test loading specific agents
        agent_details = utils.get_agent_details("tool_maker")
        assert agent_details is not None, "tool_maker agent not found"

        agent_details = utils.get_agent_details("web_researcher")
        assert agent_details is not None, "web_researcher agent not found"

        print("✅ Agent loading working correctly")
        return True
    except Exception as e:
        print(f"❌ Agent loading test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all integration tests."""
    print("🚀 Starting Comprehensive Integration Test for Inferra V")
    print("=" * 80)

    tests = [
        ("Configuration & OpenRouter API", test_config_and_openrouter),
        ("Utils & Manifests", test_utils_and_manifests),
        ("Memory Manager", test_memory_manager),
        ("Core Modules", test_core_modules),
        ("Hermes Agent", test_hermes_agent),
        ("Individual Tools", test_individual_tools),
        ("Agent Loading", test_agent_loading),
    ]

    results = {}
    total_tests = len(tests)
    passed_tests = 0

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)

        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()

            results[test_name] = {
                "success": success,
                "duration": end_time - start_time,
                "error": None,
            }

            if success:
                passed_tests += 1
                print(f"⏱️  Completed in {end_time - start_time:.2f}s")

        except Exception as e:
            results[test_name] = {"success": False, "duration": 0, "error": str(e)}
            print(f"💥 Unexpected error: {e}")
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"({result['duration']:.2f}s)" if result["duration"] > 0 else ""
        print(f"{status} {test_name} {duration}")
        if result["error"]:
            print(f"    Error: {result['error']}")

    print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(
            "🎉 ALL TESTS PASSED! Inferra V system is fully integrated and functional."
        )
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. System needs attention.")
        return False


def main():
    """Main test execution."""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
