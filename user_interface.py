#!/usr/bin/env python3
"""
Interactive User Interface for AgentK System
Provides direct communication with Hermes for planning, delegation, and error handling.
"""

import uuid
import logging
import time
import sys
import cmd
from typing import Dict, List, Optional, Any

# Import Hermes directly
from agents.hermes import hermes

# Import utilities for system health
import utils
import config


class AgentKInterface(cmd.Cmd):
    """Interactive command-line interface for AgentK System via Hermes"""

    intro = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 AGENTK SYSTEM                         ║
║              Direct Communication with Hermes               ║
╚══════════════════════════════════════════════════════════════╝

Welcome to the AgentK System!
You are now communicating directly with Hermes, the central orchestrator.
Hermes can plan, delegate tasks, reason, and fix errors automatically.

Type 'help' for available commands or just start chatting!
Examples:
  - "Create a simple web scraper"
  - "Help me analyze some data"
  - "Build a tool to convert files"
    """

    prompt = "🤖 AgentK> "

    def __init__(self):
        super().__init__()
        self.session_id = str(uuid.uuid4())
        self.conversation_active = False
        self.conversation_history = []

    def do_init(self, args):
        """Initialize the AgentK system"""
        print("🔄 Initializing AgentK System...")

        try:
            # Configure logging to reduce noise
            logging.basicConfig(level=logging.WARNING)

            # Test configuration
            if not config.default_langchain_model:
                print(
                    "⚠️  Warning: No language model configured. Some features may not work."
                )
                print("   Please ensure your API keys are set in the .env file.")

            # Test tool loading
            tools = utils.all_tool_functions()
            print(f"🛠️  Loaded {len(tools)} tools")

            # Test agent availability
            agents = utils.all_agents()
            print(f"🤖 Found {len(agents)} agents")

            print("✅ AgentK System ready!")
            print("💡 You can now start chatting with Hermes!")
            print(
                "💡 Try: 'chat' to start a conversation or just type your request directly"
            )

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback

            traceback.print_exc()

    def do_status(self, args):
        """Show system status and health"""
        print("\n📊 AGENTK SYSTEM STATUS")
        print("=" * 50)

        try:
            # Check configuration
            if config.default_langchain_model:
                print("🟢 Language Model: Configured and ready")
            else:
                print("🔴 Language Model: Not configured")

            # Check tools
            tools = utils.all_tool_functions()
            print(f"🛠️  Tools: {len(tools)} loaded")

            # Check agents
            agents = utils.all_agents()
            print(f"🤖 Agents: {len(agents)} available")

            # Session info
            print(f"🔗 Session ID: {self.session_id}")
            print(f"💬 Conversation History: {len(self.conversation_history)} messages")

            # Hermes status
            if self.conversation_active:
                print("🟢 Hermes: Active conversation")
            else:
                print("🟡 Hermes: Ready for conversation")

        except Exception as e:
            print(f"❌ Error checking status: {e}")

    def do_chat(self, args):
        """Start a conversation with Hermes"""
        print("\n💬 STARTING CONVERSATION WITH HERMES")
        print("=" * 50)
        print("You are now talking directly to Hermes, the AgentK orchestrator.")
        print(
            "Hermes will plan, delegate tasks, and coordinate other agents as needed."
        )
        print("Type 'exit' to end the conversation.\n")

        self.conversation_active = True

        try:
            # Start Hermes conversation
            print(
                "🤖 Hermes: Hello! I'm Hermes, your AgentK orchestrator. What would you like me to help you with today?"
            )

            while True:
                try:
                    user_input = input("You: ").strip()

                    if user_input.lower() in ["exit", "quit", "bye"]:
                        print(
                            "🤖 Hermes: Goodbye! Feel free to start a new conversation anytime."
                        )
                        break

                    if not user_input:
                        continue

                    # Store user input
                    self.conversation_history.append(
                        {
                            "type": "user",
                            "content": user_input,
                            "timestamp": time.time(),
                        }
                    )

                    # Communicate with Hermes
                    print("\n🤖 Hermes is thinking and planning...")
                    self._communicate_with_hermes(user_input)

                except KeyboardInterrupt:
                    print("\n🤖 Hermes: Conversation interrupted. Goodbye!")
                    break

        except Exception as e:
            print(f"❌ Error in conversation: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.conversation_active = False

    def _communicate_with_hermes(self, user_input: str):
        """Communicate directly with Hermes agent"""
        try:
            # Create a new session for this interaction
            session_id = f"{self.session_id}_{int(time.time())}"

            # Call Hermes directly - this will handle the full conversation flow
            # Hermes will plan, delegate, reason, and handle errors automatically
            result = hermes(session_id)

            # Store the interaction
            self.conversation_history.append(
                {
                    "type": "hermes",
                    "content": "Conversation completed",
                    "session_id": session_id,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            print(f"❌ Error communicating with Hermes: {e}")
            print("🔧 Hermes will attempt to handle this error and continue...")

    def do_quick_start(self, args):
        """Run a quick demonstration with Hermes"""
        print("🚀 QUICK START DEMO WITH HERMES")
        print("=" * 40)
        print(
            "This will start a conversation with Hermes and demonstrate its capabilities.\n"
        )

        # Start a demo conversation
        self.conversation_active = True

        try:
            print("🤖 Starting demo conversation with Hermes...")
            print("📝 Demo task: 'Create a simple hello world script'\n")

            # Create a demo session
            demo_session_id = f"demo_{self.session_id}_{int(time.time())}"

            print("🤖 Hermes: I'll help you create a simple hello world script!")
            print(
                "🤖 Hermes: Let me plan this task and delegate it to the appropriate agents..."
            )

            # Call Hermes with a demo task
            result = hermes(demo_session_id)

            print("\n✅ Demo completed!")
            print(
                "💡 You can now try 'chat' to start your own conversation with Hermes"
            )

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.conversation_active = False

    def do_help(self, args):
        """Show help information"""
        print("\n🔧 AVAILABLE COMMANDS")
        print("=" * 50)

        commands = {
            "init": "Initialize the AgentK system",
            "status": "Show system status and health",
            "chat": "Start a conversation with Hermes",
            "quick_start": "Run a demonstration with Hermes",
            "exit": "Exit the interface",
        }

        for cmd, desc in commands.items():
            print(f"🔹 {cmd:<15} - {desc}")

        print("\n💡 TIPS:")
        print("  • Start with 'init' to initialize the system")
        print("  • Try 'quick_start' for a demonstration")
        print("  • Use 'chat' to talk directly with Hermes")
        print("  • You can also just type your request directly!")

        print("\n🤖 ABOUT HERMES:")
        print("  Hermes is the central orchestrator that can:")
        print("  • Understand your goals and create detailed plans")
        print("  • Delegate tasks to specialized agents")
        print("  • Create new agents and tools as needed")
        print("  • Handle errors and adapt plans automatically")
        print("  • Coordinate complex multi-step workflows")

    def do_exit(self, args):
        """Exit the interface"""
        print("👋 Goodbye!")
        return True

    def default(self, line):
        """Handle unrecognized commands as direct communication with Hermes"""
        if not line.strip():
            return

        print(f"\n🤖 Interpreting '{line}' as a request for Hermes...")
        print("🔄 Starting conversation with Hermes...\n")

        # Treat any unrecognized input as a direct request to Hermes
        self.conversation_active = True

        try:
            # Store user input
            self.conversation_history.append(
                {"type": "user", "content": line, "timestamp": time.time()}
            )

            print(
                "🤖 Hermes: I understand your request. Let me work on that for you..."
            )

            # Communicate with Hermes
            self._communicate_with_hermes(line)

            print("\n💡 If you want to continue the conversation, use 'chat' command")

        except Exception as e:
            print(f"❌ Error processing request: {e}")
            print("💡 Try using 'chat' command for a proper conversation with Hermes")
        finally:
            self.conversation_active = False


def main():
    """Main function to start the interactive interface"""
    try:
        interface = AgentKInterface()
        interface.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Interface error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
