#!/usr/bin/env python3
"""
Inferra V System Startup Script

This script provides a simple way to start and verify the Inferra V system.
"""

import sys
import os
import time
from typing import Optional


def check_system_health() -> bool:
    """Perform basic system health checks."""
    print("🔍 Performing system health checks...")

    try:
        # Check configuration
        import config

        if config.default_langchain_model is None:
            print("❌ Language model not initialized")
            return False
        print("✅ Configuration loaded successfully")

        # Check tools
        import utils

        tools = utils.all_tool_functions()
        if len(tools) < 5:
            print(f"⚠️  Only {len(tools)} tools loaded (expected 7+)")
        else:
            print(f"✅ {len(tools)} tools loaded successfully")

        # Check agents
        agents = utils.all_agents()
        if len(agents) < 3:
            print(f"⚠️  Only {len(agents)} agents available (expected 4+)")
        else:
            print(f"✅ {len(agents)} agents available")

        # Check memory manager
        import memory_manager

        mm = memory_manager.MemoryManager()
        mm.close()
        print("✅ Memory manager operational")

        return True

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def start_hermes_interactive():
    """Start Hermes in interactive mode."""
    try:
        from agents.hermes import hermes
        import uuid

        print("\n🚀 Starting Hermes Interactive Mode...")
        print("=" * 50)

        # Generate a unique session ID
        session_id = str(uuid.uuid4())[:8]

        # Start Hermes
        hermes(session_id)

    except KeyboardInterrupt:
        print("\n👋 Session ended by user")
    except Exception as e:
        print(f"❌ Failed to start Hermes: {e}")
        return False

    return True


def show_system_info():
    """Display system information."""
    print("\n📊 Inferra V System Information")
    print("=" * 50)

    try:
        import config

        print(f"🤖 Model: {config.config.model.model_name}")
        print(f"🔗 Provider: {config.config.model.provider}")
        print(f"🌡️  Temperature: {config.config.model.temperature}")

        import utils

        tools = utils.all_tool_functions()
        agents = utils.all_agents()

        print(f"🛠️  Tools Available: {len(tools)}")
        print(f"👥 Agents Available: {len(agents)}")

        print(f"\n🔧 Available Tools:")
        for i, tool in enumerate(tools, 1):
            tool_name = getattr(tool, "name", f"Tool_{i}")
            print(f"  {i}. {tool_name}")

        print(f"\n🤖 Available Agents:")
        for i, (agent_name, description) in enumerate(agents.items(), 1):
            print(f"  {i}. {agent_name}: {description[:60]}...")

    except Exception as e:
        print(f"❌ Failed to get system info: {e}")


def main():
    """Main startup function."""
    print("🌟 Welcome to Inferra V - Advanced AI Agent System")
    print("=" * 60)

    # Perform health checks
    if not check_system_health():
        print("\n❌ System health checks failed. Please review the issues above.")
        sys.exit(1)

    print("\n✅ All health checks passed!")

    # Show system info
    show_system_info()

    # Interactive menu
    while True:
        print("\n" + "=" * 60)
        print("🎯 What would you like to do?")
        print("1. Start Hermes Interactive Session")
        print("2. Run System Health Check")
        print("3. Show System Information")
        print("4. Exit")
        print("=" * 60)

        try:
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                start_hermes_interactive()
            elif choice == "2":
                check_system_health()
            elif choice == "3":
                show_system_info()
            elif choice == "4":
                print("\n👋 Goodbye! Thank you for using Inferra V.")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using Inferra V.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
