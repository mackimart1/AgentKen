from langchain_core.tools import tool
import utils


@tool
def list_available_agents():
    """Lists available agents and their descriptions based on the agents manifest."""
    # The refactored utils.all_agents now reads from the manifest
    return utils.all_agents()
