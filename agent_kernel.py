import warnings
import sys  # Import sys
import os  # Import os

# --- Add project root to sys.path ---
# Get the absolute path of the directory containing this script (agent_kernel.py)
project_root = os.path.dirname(os.path.abspath(__file__))
# Add it to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")
# --- End sys.path modification ---

# --- Pre-import key modules ---
try:
    import memory_manager  # Attempt to import early
    import utils  # Ensure utils is loaded

    print("Successfully pre-imported memory_manager and utils.")
except ImportError as e:
    print(f"ERROR: Failed to pre-import core modules: {e}")
    # Decide if we should exit or continue with potential errors
    sys.exit(f"Exiting due to critical import failure: {e}")
# --- End pre-import ---


from dotenv import load_dotenv

load_dotenv()

# Suppress specific UserWarnings from langchain (if needed)
# Note: This might hide other potentially useful warnings from this package.
# warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

from agents import hermes
from uuid import uuid4

uuid = str(uuid4())
hermes.hermes(uuid)
