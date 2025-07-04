"""
Handles centralized configuration for the entire agent system.

Reads environment variables and sets up configurations for:
- Language Models (OpenRouter, OpenAI, Anthropic, Ollama)
- Orchestrator Settings
- Agent Behaviors
- Tool Configurations

Provides a single, unified point of configuration to simplify management and deployment.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Environment Variable Loading ---
def get_env_var(name: str, default: Any = None) -> Any:
    """Get an environment variable with a default value."""
    return os.getenv(name, default)


# --- Language Model Configuration ---
class ModelConfig:
    """Configuration for the language model."""

    def __init__(self):
        self.provider = get_env_var("DEFAULT_MODEL_PROVIDER", "OPENROUTER").upper()
        self.model_name = get_env_var(
            "DEFAULT_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free"
        )
        self.temperature = float(get_env_var("DEFAULT_MODEL_TEMPERATURE", "0.1"))
        self.api_key = get_env_var(
            "OPENROUTER_API_KEY",
            "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",
        )
        self.base_url = get_env_var(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }


# --- Orchestrator Configuration ---
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    def __init__(self):
        self.max_concurrent_tasks = int(
            get_env_var("ORCHESTRATOR_MAX_CONCURRENT_TASKS", "20")
        )
        self.default_task_priority = get_env_var(
            "ORCHESTRATOR_DEFAULT_PRIORITY", "NORMAL"
        )
        self.enable_optimization = (
            get_env_var("ORCHESTRATOR_ENABLE_OPTIMIZATION", "true").lower() == "true"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "default_task_priority": self.default_task_priority,
            "enable_optimization": self.enable_optimization,
        }


# --- Agent Configuration ---
class AgentConfig:
    """Configuration for agents."""

    def __init__(self):
        self.heartbeat_interval = int(get_env_var("AGENT_HEARTBEAT_INTERVAL", "30"))
        self.max_retries = int(get_env_var("AGENT_MAX_RETRIES", "3"))
        self.default_timeout = int(get_env_var("AGENT_DEFAULT_TIMEOUT", "60"))

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return {
            "heartbeat_interval": self.heartbeat_interval,
            "max_retries": self.max_retries,
            "default_timeout": self.default_timeout,
        }


# --- Tool Configuration ---
class ToolConfig:
    """Configuration for tools."""

    def __init__(self):
        self.default_timeout = int(get_env_var("TOOL_DEFAULT_TIMEOUT", "30"))
        self.default_rate_limit = int(
            get_env_var("TOOL_DEFAULT_RATE_LIMIT", "10")
        )  # per second

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return {
            "default_timeout": self.default_timeout,
            "default_rate_limit": self.default_rate_limit,
        }


# --- Central Configuration Class ---
class AppConfig:
    """Centralized configuration for the entire application."""

    def __init__(self):
        self.model = ModelConfig()
        self.orchestrator = OrchestratorConfig()
        self.agent = AgentConfig()
        self.tool = ToolConfig()
        self.log_level = get_env_var("LOG_LEVEL", "INFO").upper()

    def log_configurations(self):
        """Log the current configurations."""
        logging.info("--- Application Configuration ---")
        logging.info(f"Log Level: {self.log_level}")
        logging.info(f"Model Config: {self.model.to_dict()}")
        logging.info(f"Orchestrator Config: {self.orchestrator.to_dict()}")
        logging.info(f"Agent Config: {self.agent.to_dict()}")
        logging.info(f"Tool Config: {self.tool.to_dict()}")
        logging.info("---------------------------------")


# --- Language Model Initialization ---
def initialize_language_model():
    """Initialize the default language model based on configuration."""
    try:
        # Initialize OpenRouter model
        from langchain_openai import ChatOpenAI

        # Get model configuration
        model_config = config.model

        if (
            model_config.api_key
            and model_config.api_key != "your_openrouter_api_key_here"
        ):
            model = ChatOpenAI(
                model=model_config.model_name,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                temperature=model_config.temperature,
            )
            logging.info(f"Initialized OpenRouter model: {model_config.model_name}")
            return model
        else:
            logging.warning(
                "OpenRouter API key not found, language model not initialized"
            )
            return None

    except ImportError as e:
        logging.warning(f"Could not import OpenRouter model: {e}")
        return None
    except Exception as e:
        logging.error(f"Error initializing language model: {e}")
        return None


def reinitialize_openrouter_model(new_api_key: str, model_name: str = None):
    """Reinitialize the OpenRouter model with a new API key."""
    global default_langchain_model
    try:
        from langchain_openai import ChatOpenAI

        # Use provided model name or default from config
        model_to_use = model_name or config.model.model_name

        default_langchain_model = ChatOpenAI(
            model=model_to_use,
            api_key=new_api_key,
            base_url=config.model.base_url,
            temperature=config.model.temperature,
        )
        logging.info(f"Successfully reinitialized OpenRouter model: {model_to_use}")

    except Exception as e:
        logging.error(f"Failed to reinitialize OpenRouter model: {e}")
        raise e


# --- Global Configuration Instance ---
# Create a single instance of the AppConfig to be used throughout the application.
config = AppConfig()

# Initialize the default language model
default_langchain_model = initialize_language_model()

# --- Example of how to use the config ---
if __name__ == "__main__":
    # Log the configurations on startup
    config.log_configurations()

    # Access configurations from other modules like this:
    # from config import config
    # print(f"Using model provider: {config.model.provider}")
    # print(f"Orchestrator max tasks: {config.orchestrator.max_concurrent_tasks}")
