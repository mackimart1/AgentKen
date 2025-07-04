#!/usr/bin/env python3
"""
Test script for AI communication functionality
"""

import sys
import logging
from core.ai_communication import AIAssistant, test_ai_communication

def main():
    """Test AI communication"""
    print("🧪 Testing AI Communication with DeepSeek")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test basic functionality
        print("1️⃣ Testing basic AI communication...")
        success = test_ai_communication()
        
        if success:
            print("✅ AI communication test passed!")
            
            # Test interactive conversation
            print("\n2️⃣ Testing interactive conversation...")
            assistant = AIAssistant()
            
            test_messages = [
                "Hello, how are you?",
                "What can you help me with?",
                "What's the system status?",
                "How do I create a plan?",
                "Can you optimize my workflow?"
            ]
            
            for message in test_messages:
                print(f"\n👤 User: {message}")
                try:
                    response = assistant.chat(message)
                    print(f"🤖 AI: {response}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("\n✅ All tests completed!")
            
        else:
            print("❌ AI communication test failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())