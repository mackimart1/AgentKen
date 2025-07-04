#!/usr/bin/env python3
"""
Simple User Interface for AgentK System
Direct communication with Hermes without complex dependencies.
"""

import uuid
import logging
import time
import sys

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)


def simple_hermes_interface():
    """Simple interface that directly calls Hermes without complex imports"""

    print(
        """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 AGENTK SYSTEM                         ║
║              Direct Communication with Hermes               ║
╚═════════════════════════════��════════════════════════════════╝

Welcome to the AgentK System!
You are now communicating directly with Hermes, the central orchestrator.
Hermes can plan, delegate tasks, reason, and fix errors automatically.

Type 'exit' to end the conversation.
    """
    )

    # Generate a unique session ID
    session_id = str(uuid.uuid4())

    try:
        # Import Hermes function directly
        from agents.hermes import hermes

        print("🤖 Hermes: Hello! I'm Hermes, your AgentK orchestrator.")
        print("🤖 Hermes: What would you like me to help you with today?")

        # Start the Hermes conversation loop
        # This will handle all the interaction automatically
        result = hermes(session_id)

        print("\n✅ Conversation completed!")

    except ImportError as e:
        print(f"❌ Error importing Hermes: {e}")
        print("💡 This might be due to missing dependencies.")
        print("💡 Please check that all required packages are installed.")

    except Exception as e:
        print(f"❌ Error during conversation: {e}")
        print("🔧 Hermes encountered an error but will attempt to handle it.")
        import traceback

        traceback.print_exc()


def main():
    """Main function to start the simple interface"""
    try:
        simple_hermes_interface()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Interface error: {e}")


if __name__ == "__main__":
    main()
