#!/usr/bin/env python3
"""
Test OpenRouter API key and connection
"""

import requests
import json


def test_openrouter_api():
    """Test the OpenRouter API key"""

    api_key = (
        "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"
    )
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-chat-v3-0324:free"

    print("🔑 Testing OpenRouter API Key")
    print("=" * 40)
    print(f"API Key: {api_key[:20]}...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://inferra-v.local",
        "X-Title": "Inferra V Enhanced System",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Hello, please respond with 'API test successful'",
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50,
    }

    try:
        print("\n📡 Making API request...")
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                ai_response = result["choices"][0]["message"]["content"]
                print(f"✅ API Response: {ai_response}")
                return True
            else:
                print(f"❌ Invalid response format: {result}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False


if __name__ == "__main__":
    success = test_openrouter_api()
    if success:
        print("\n🎉 API test successful!")
    else:
        print("\n💡 Suggestions:")
        print("1. Check if the API key is valid")
        print("2. Verify internet connection")
        print("3. Check if the model name is correct")
        print("4. Try a different model if this one is unavailable")
