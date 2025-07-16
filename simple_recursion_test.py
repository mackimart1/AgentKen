#!/usr/bin/env python3
"""
Simple test to verify agent_smith recursion fix.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import():
    """Test that agent_smith can be imported without errors."""
    try:
        print("Testing agent_smith import...")
        from agents.agent_smith import agent_smith, creation_state, workflow_manager
        print("✅ Successfully imported agent_smith")
        
        # Test state initialization
        print(f"✅ Creation state initialized: {creation_state.max_reasoning_iterations} max iterations")
        
        # Test workflow manager
        print(f"✅ Workflow manager initialized: {workflow_manager is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_loop_detection_config():
    """Test loop detection configuration."""
    try:
        from agents.agent_smith import creation_state
        
        print("Testing loop detection configuration...")
        print(f"   Max reasoning iterations: {creation_state.max_reasoning_iterations}")
        print(f"   Max repeated content: {creation_state.max_repeated_content}")
        print(f"   Max duration: {creation_state.max_duration} seconds")
        print(f"   Max inactivity: {creation_state.max_inactivity} seconds")
        
        # Verify reasonable limits
        assert creation_state.max_reasoning_iterations > 25, "Reasoning iterations should be > 25"
        assert creation_state.max_repeated_content >= 3, "Repeated content limit should be >= 3"
        
        print("✅ Loop detection properly configured")
        return True
    except Exception as e:
        print(f"❌ Loop detection test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🧪 Simple Agent Smith Recursion Fix Test")
    print("=" * 50)
    
    tests = [test_import, test_loop_detection_config]
    passed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ Basic recursion fix verification successful!")
        print("\n🎯 Key improvements verified:")
        print("   • Agent Smith imports without errors")
        print("   • Loop detection mechanisms configured")
        print("   • Reasonable iteration limits set")
    else:
        print("❌ Some tests failed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)