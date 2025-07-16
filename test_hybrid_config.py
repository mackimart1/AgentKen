#!/usr/bin/env python3
"""
Test script to verify the hybrid model configuration is working properly.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from agents.hermes import hermes
import uuid

def test_hybrid_configuration():
    """Test that the hybrid configuration is working properly."""
    print("=" * 60)
    print("Testing Hybrid Model Configuration")
    print("=" * 60)
    
    # Test 1: Check model initialization
    print("\n1. Testing Model Initialization:")
    tool_model = config.get_model_for_tools()
    chat_model = config.get_model_for_chat()
    
    print(f"   Tool Model: {type(tool_model).__name__}")
    print(f"   Chat Model: {type(chat_model).__name__}")
    
    if tool_model is None:
        print("   ❌ Tool model not initialized!")
        return False
    
    if chat_model is None:
        print("   ❌ Chat model not initialized!")
        return False
    
    print("   ✅ Both models initialized successfully")
    
    # Test 2: Check model types
    print("\n2. Testing Model Types:")
    expected_tool_model = "ChatGoogleGenerativeAI"
    expected_chat_model = "ChatOpenAI"
    
    if type(tool_model).__name__ == expected_tool_model:
        print(f"   ✅ Tool model is correct: {expected_tool_model}")
    else:
        print(f"   ❌ Tool model is wrong: expected {expected_tool_model}, got {type(tool_model).__name__}")
        return False
    
    if type(chat_model).__name__ == expected_chat_model:
        print(f"   ✅ Chat model is correct: {expected_chat_model}")
    else:
        print(f"   ❌ Chat model is wrong: expected {expected_chat_model}, got {type(chat_model).__name__}")
        return False
    
    # Test 3: Test simple agent execution
    print("\n3. Testing Agent Execution with Hybrid Models:")
    try:
        print("   Running a simple test with Hermes agent...")
        test_uuid = str(uuid.uuid4())
        
        # This should use the hybrid configuration
        print("   Note: This test will prompt for input. Type 'exit' to end the test.")
        print("   You can test with a simple request like 'list available agents'")
        
        # Uncomment the line below to run the actual test
        # result = hermes(test_uuid)
        
        print("   ✅ Agent execution test skipped (requires user interaction)")
        print("   To test manually, uncomment the hermes() call in the script")
        
    except Exception as e:
        print(f"   ❌ Agent execution failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Configuration Test PASSED")
    print("=" * 60)
    print("\nConfiguration Summary:")
    print(f"• Tool Calling: Google Gemini ({config.config.model.google_model_name})")
    print(f"• Chat Operations: OpenRouter ({config.config.model.model_name})")
    print(f"• Tool Provider: {config.config.model.tool_calling_provider}")
    print(f"• Chat Provider: {config.config.model.chat_provider}")
    
    return True

if __name__ == "__main__":
    success = test_hybrid_configuration()
    sys.exit(0 if success else 1)