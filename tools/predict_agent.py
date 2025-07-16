import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import joblib
import logging
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "agent_selector_model.keras"
ENCODER_PATH = "agent_selector_label_encoder.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Must match training script

# --- Load Models ---
agent_selector_model = None
label_encoder = None
embedding_model = None

try:
    if os.path.exists(MODEL_PATH):
        agent_selector_model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Loaded agent selector model from {MODEL_PATH}")
    else:
        logger.warning(
            f"Agent selector model not found at {MODEL_PATH}. Prediction tool will be disabled."
        )

    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
        logger.info(f"Loaded label encoder from {ENCODER_PATH}")
    else:
        logger.warning(
            f"Label encoder not found at {ENCODER_PATH}. Prediction tool will be disabled."
        )
        agent_selector_model = None  # Disable model if encoder is missing

    if (
        agent_selector_model and label_encoder
    ):  # Only load embedder if model/encoder loaded
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded for prediction.")

except Exception as e:
    logger.error(f"Error loading models/encoders for predict_agent: {e}", exc_info=True)
    agent_selector_model = None
    label_encoder = None
    embedding_model = None

# --- Tool Definition ---


class PredictAgentInput(BaseModel):
    task_description: str = Field(
        description="The description of the task to be assigned."
    )


@tool(args_schema=PredictAgentInput)
def predict_agent(task_description: str) -> Dict[str, float]:
    """
    Predicts the probability distribution of the best agent for a given task
    based on a trained neural network model. Returns an empty dictionary if
    the model is not available or prediction fails.
    """
    if not agent_selector_model or not label_encoder or not embedding_model:
        logger.warning("Agent prediction model/encoder/embedder not available.")
        return {}  # Return empty if model/encoder/embedder isn't loaded

    if not task_description or not task_description.strip():
        logger.warning("predict_agent called with empty task description.")
        return {}

    try:
        # 1. Embed the task description
        task_embedding = embedding_model.encode(
            [task_description]
        )  # Encode expects a list

        # 2. Predict probabilities
        predictions = agent_selector_model.predict(task_embedding)
        probabilities = predictions[0]  # Get probabilities for the first (only) input

        # 3. Map probabilities to agent names
        agent_probabilities = {
            agent_name: float(prob)  # Ensure float type for JSON serialization
            for agent_name, prob in zip(label_encoder.classes_, probabilities)
        }

        logger.info(
            f"Predicted agent probabilities for task '{task_description[:50]}...': {agent_probabilities}"
        )
        return agent_probabilities

    except Exception as e:
        logger.error(f"Error during agent prediction: {e}", exc_info=True)
        return {}  # Return empty on error


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing predict_agent tool...")
    if agent_selector_model and label_encoder and embedding_model:
        test_task = "Create a tool to summarize web pages"
        print(f"\nPredicting for task: '{test_task}'")
        probs = predict_agent(task_description=test_task)
        if probs:
            # Sort by probability descending for display
            sorted_probs = dict(
                sorted(probs.items(), key=lambda item: item[1], reverse=True)
            )
            print("Probabilities:")
            for agent, prob in sorted_probs.items():
                print(f"- {agent}: {prob:.4f}")
        else:
            print("Prediction failed or model not loaded.")

        test_task_2 = "Research the latest advancements in AI"
        print(f"\nPredicting for task: '{test_task_2}'")
        probs_2 = predict_agent(task_description=test_task_2)
        if probs_2:
            sorted_probs_2 = dict(
                sorted(probs_2.items(), key=lambda item: item[1], reverse=True)
            )
            print("Probabilities:")
            for agent, prob in sorted_probs_2.items():
                print(f"- {agent}: {prob:.4f}")
        else:
            print("Prediction failed or model not loaded.")
    else:
        print("Agent prediction model/encoder/embedder not loaded. Cannot run test.")
