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
        
        # OpenRouter configuration
        self.api_key = get_env_var(
            "OPENROUTER_API_KEY",
            "API-KEY",
        )
        self.base_url = get_env_var(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        
        # Google Gemini configuration
        self.google_api_key = get_env_var("GOOGLE_API_KEY")
        self.google_model_name = get_env_var("GOOGLE_MODEL_NAME", "gemini-2.0-flash-exp")
        
        # Hybrid configuration
        self.tool_calling_provider = get_env_var("TOOL_CALLING_PROVIDER", "GOOGLE").upper()
        self.chat_provider = get_env_var("CHAT_PROVIDER", "OPENROUTER").upper()

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "google_api_key": self.google_api_key,
            "google_model_name": self.google_model_name,
            "tool_calling_provider": self.tool_calling_provider,
            "chat_provider": self.chat_provider,
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


def initialize_google_gemini_model():
    """Initialize Google Gemini model for tool calling."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        model_config = config.model
        
        if model_config.google_api_key:
            model = ChatGoogleGenerativeAI(
                model=model_config.google_model_name,
                google_api_key=model_config.google_api_key,
                temperature=model_config.temperature,
            )
            logging.info(f"Initialized Google Gemini model: {model_config.google_model_name}")
            return model
        else:
            logging.warning("Google API key not found, Gemini model not initialized")
            return None
            
    except ImportError as e:
        logging.warning(f"Could not import Google Gemini model: {e}")
        return None
    except Exception as e:
        logging.error(f"Error initializing Google Gemini model: {e}")
        return None


class HybridModelManager:
    """Manages hybrid model configuration with Google Gemini for tools and OpenRouter for chat."""
    
    def __init__(self):
        self.openrouter_model = None
        self.gemini_model = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize both OpenRouter and Google Gemini models."""
        # Initialize OpenRouter model for chat
        self.openrouter_model = initialize_language_model()
        
        # Initialize Google Gemini model for tool calling
        self.gemini_model = initialize_google_gemini_model()
        
        if self.openrouter_model and self.gemini_model:
            logging.info("Hybrid model setup complete: OpenRouter for chat, Google Gemini for tools")
        elif self.openrouter_model:
            logging.warning("Only OpenRouter model available")
        elif self.gemini_model:
            logging.warning("Only Google Gemini model available")
        else:
            logging.error("No models available!")
    
    def get_tool_model(self):
        """Get the model for tool calling (Google Gemini preferred)."""
        if config.model.tool_calling_provider == "GOOGLE" and self.gemini_model:
            return self.gemini_model
        elif self.openrouter_model:
            logging.warning("Using OpenRouter for tools (Google Gemini not available)")
            return self.openrouter_model
        else:
            logging.error("No model available for tool calling!")
            return None
    
    def get_chat_model(self):
        """Get the model for chat (OpenRouter preferred)."""
        if config.model.chat_provider == "OPENROUTER" and self.openrouter_model:
            return self.openrouter_model
        elif self.gemini_model:
            logging.warning("Using Google Gemini for chat (OpenRouter not available)")
            return self.gemini_model
        else:
            logging.error("No model available for chat!")
            return None
    
    def get_model_for_task(self, task_type: str = "chat"):
        """Get the appropriate model based on task type."""
        if task_type.lower() in ["tool", "tools", "function", "function_calling"]:
            return self.get_tool_model()
        else:
            return self.get_chat_model()


def reinitialize_openrouter_model(new_api_key: str, model_name: str = None):
    """Reinitialize the OpenRouter model with a new API key."""
    global default_langchain_model, hybrid_model_manager
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
        
        # Reinitialize hybrid manager
        hybrid_model_manager = HybridModelManager()

    except Exception as e:
        logging.error(f"Failed to reinitialize OpenRouter model: {e}")
        raise e





# --- Global Configuration Instance ---
# Create a single instance of the AppConfig to be used throughout the application.
config = AppConfig()

# Initialize the default language model
default_langchain_model = initialize_language_model()

# Initialize the hybrid model manager
hybrid_model_manager = HybridModelManager()

# --- Example of how to use the config ---
if __name__ == "__main__":
    # Log the configurations on startup
    config.log_configurations()
    
    # Test hybrid model setup
    print("\n--- Hybrid Model Setup ---")
    print(f"Tool calling provider: {config.model.tool_calling_provider}")
    print(f"Chat provider: {config.model.chat_provider}")
    print(f"Google Gemini model: {config.model.google_model_name}")
    print(f"OpenRouter model: {config.model.model_name}")
    
    # Test model access
    tool_model = hybrid_model_manager.get_tool_model()
    chat_model = hybrid_model_manager.get_chat_model()
    print(f"Tool model available: {tool_model is not None}")
    print(f"Chat model available: {chat_model is not None}")

    # Access configurations from other modules like this:
    # from config import config, hybrid_model_manager
    # tool_model = hybrid_model_manager.get_tool_model()
    # chat_model = hybrid_model_manager.get_chat_model()


# --- Convenience Functions ---
def get_model_for_tools():
    """Convenience function to get the model for tool calling."""
    return hybrid_model_manager.get_tool_model()


def get_model_for_chat():
    """Convenience function to get the model for chat."""
    return hybrid_model_manager.get_chat_model()


def get_model(task_type: str = "chat"):
    """Convenience function to get the appropriate model based on task type."""
    return hybrid_model_manager.get_model_for_task(task_type)
