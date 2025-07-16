#!/usr/bin/env python3
"""
Test script to verify OpenRouter API configuration and connectivity.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import config
    from langchain_core.messages import HumanMessage

    def test_openrouter_connection():
        """Test OpenRouter API connection and model initialization."""
        print("🧪 Testing OpenRouter Configuration...")
        print("=" * 50)

        # Check configuration
        print(f"Provider: {config.config.model.provider}")
        print(f"Model: {config.config.model.model_name}")
        print(f"Base URL: {config.config.model.base_url}")
        print(
            f"API Key: {'✅ Set' if config.config.model.api_key and config.config.model.api_key != 'your_openrouter_api_key_here' else '❌ Not Set'}"
        )
        print(f"Temperature: {config.config.model.temperature}")
        print()

        # Test model initialization
        if config.default_langchain_model is None:
            print("❌ Model not initialized")
            return False

        print("✅ Model initialized successfully")
        print(f"Model type: {type(config.default_langchain_model).__name__}")
        print()

        # Test simple API call
        try:
            print("🔄 Testing API call...")
            test_message = HumanMessage(
                content="Hello! Please respond with 'OpenRouter is working correctly.'"
            )
            response = config.default_langchain_model.invoke([test_message])

            print("✅ API call successful!")
            print(f"Response: {response.content}")
            print()

            return True

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return False

    def main():
        """Main test function."""
        print("🚀 OpenRouter Configuration Test")
        print("=" * 50)

        success = test_openrouter_connection()

        print("=" * 50)
        if success:
            print("🎉 All tests passed! OpenRouter is configured correctly.")
        else:
            print("💥 Tests failed. Please check your configuration.")
            print("\nTroubleshooting:")
            print("1. Verify your OpenRouter API key is correct")
            print("2. Check your internet connection")
            print("3. Ensure the model name is valid")
            print("4. Check OpenRouter service status")

        return success

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
