#!/usr/bin/env python3
"""
Script to fix Google API references in meta_agent.py
"""


def fix_meta_agent():
    """Fix Google API references in meta_agent.py"""
    file_path = "agents/meta_agent.py"

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace Google API imports
    content = content.replace(
        "from google.api_core.exceptions import ResourceExhausted",
        "# OpenRouter uses standard HTTP errors, not Google-specific exceptions\nfrom requests.exceptions import HTTPError, RequestException",
    )

    # Replace ResourceExhausted with HTTPError, RequestException
    content = content.replace(
        "except ResourceExhausted as e:", "except (HTTPError, RequestException) as e:"
    )

    # Replace Google API messages
    content = content.replace("Google API Quota Exceeded", "OpenRouter API Error")

    content = content.replace("Google API Key", "OpenRouter API Key")

    # Replace reinitialize function call
    content = content.replace(
        "config.reinitialize_google_model(new_key)",
        "config.reinitialize_openrouter_model(new_key)",
    )

    # Write the file back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Fixed meta_agent.py - replaced Google API references with OpenRouter")


if __name__ == "__main__":
    fix_meta_agent()
