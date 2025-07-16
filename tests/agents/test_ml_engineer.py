# tests/agents/test_ml_engineer.py
import unittest
import os
import sys
import importlib

# Add the parent directory to sys.path to allow imports from agents and tools
# Ensure this runs correctly regardless of where unittest is invoked from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure the 'agents' directory is also findable if needed directly
agents_dir = os.path.join(project_root, "agents")
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)


class TestMLEngineerAgent(unittest.TestCase):

    AGENT_MODULE_PATH = "agents.ml_engineer"
    AGENT_FILE_PATH = os.path.join(project_root, "agents", "ml_engineer.py")

    @classmethod
    def setUpClass(cls):
        """Ensure the agent module can be imported before running tests."""
        try:
            cls.agent_module = importlib.import_module(cls.AGENT_MODULE_PATH)
            cls.ml_engineer = getattr(cls.agent_module, "ml_engineer")
        except ImportError as e:
            raise unittest.SkipTest(
                f"Skipping tests: Failed to import agent module {cls.AGENT_MODULE_PATH}: {e}"
            )
        except AttributeError:
            raise unittest.SkipTest(
                f"Skipping tests: Agent function 'ml_engineer' not found in {cls.AGENT_MODULE_PATH}"
            )
        except Exception as e:
            raise unittest.SkipTest(
                f"Skipping tests: Error importing agent {cls.AGENT_MODULE_PATH}: {e}"
            )

    def test_agent_file_exists(self):
        """Check if the agent file exists."""
        self.assertTrue(
            os.path.exists(self.AGENT_FILE_PATH),
            f"Agent file does not exist: {self.AGENT_FILE_PATH}",
        )

    def test_agent_importable_and_callable(self):
        """Check if the agent function is callable (already checked in setUpClass)."""
        self.assertTrue(
            callable(self.ml_engineer), "ml_engineer function is not callable."
        )

    def test_basic_invocation(self):
        """Perform a basic invocation to check for runtime errors in the graph definition."""
        try:
            initial_state = {"messages": [("human", "Test task for basic invocation.")]}
            # We expect this to run through the simulated steps
            final_state = self.ml_engineer(initial_state)

            self.assertIsInstance(final_state, dict)
            self.assertIn("messages", final_state)
            messages = final_state.get("messages", [])
            self.assertTrue(isinstance(messages, list), "Messages should be a list.")
            # Check if the agent produced some output messages beyond the initial human one
            self.assertTrue(
                len(messages) > 1, "Agent did not produce sufficient output messages."
            )

            # Check if the final message indicates completion or error (as expected by the simple graph)
            last_message = messages[-1] if messages else None
            self.assertIsNotNone(last_message, "No messages found in final state.")
            self.assertIsInstance(
                last_message, tuple, f"Last message is not a tuple: {last_message}"
            )
            last_role, last_msg = last_message
            self.assertIn(
                last_role,
                ["ai", "function"],
                f"Unexpected role in last message: {last_role}",
            )
            self.assertTrue(
                "Finished execution simulation." in last_msg
                or "error occurred" in last_msg,
                f"Final message does not indicate expected completion or error: {last_msg}",
            )

            print("\n--- Basic Invocation Test Output --- (test_basic_invocation)")
            for msg in messages:
                role = msg[0] if isinstance(msg, tuple) else "unknown"
                content = (
                    msg[1] if isinstance(msg, tuple) and len(msg) > 1 else str(msg)
                )
                print(f"{role.capitalize()}: {content}")
            print("--- End Basic Invocation Test Output --- (test_basic_invocation)")

        except Exception as e:
            # Print the exception for debugging before failing
            import traceback

            traceback.print_exc()
            self.fail(f"Agent invocation failed with an unexpected error: {e}")


if __name__ == "__main__":
    unittest.main()
