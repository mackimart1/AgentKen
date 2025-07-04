import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the LabelEncoder
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LOG_FILE = "agent_assignment_log.csv"
MODEL_SAVE_PATH = "agent_selector_model.keras"
ENCODER_SAVE_PATH = "agent_selector_label_encoder.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Must match memory_manager
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 10  # Adjust as needed
BATCH_SIZE = 8  # Adjust as needed

# --- Functions ---


def load_data(filepath: str) -> pd.DataFrame:
    """Loads the agent assignment log data."""
    if not os.path.exists(filepath):
        logger.error(f"Training data file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        # Basic validation
        if "task" not in df.columns or "agent" not in df.columns:
            logger.error(f"Missing required columns ('task', 'agent') in {filepath}")
            return None
        df.dropna(
            subset=["task", "agent"], inplace=True
        )  # Drop rows with missing crucial data
        logger.info(f"Using {len(df)} valid records for training.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}", exc_info=True)
        return None


def preprocess_data(
    df: pd.DataFrame, embedding_model: SentenceTransformer, encoder_save_path: str
):
    """Encodes tasks using embeddings and agents using LabelEncoder."""
    if df is None or df.empty:
        return None, None, None

    # Embed task descriptions
    logger.info("Generating task embeddings...")
    task_embeddings = embedding_model.encode(
        df["task"].tolist(), show_progress_bar=True
    )

    # Encode agent labels
    logger.info("Encoding agent labels...")
    label_encoder = LabelEncoder()
    agent_labels_encoded = label_encoder.fit_transform(df["agent"])
    num_classes = len(label_encoder.classes_)
    logger.info(f"Found {num_classes} unique agent classes: {label_encoder.classes_}")

    # Save the label encoder for later use during prediction
    try:
        joblib.dump(label_encoder, encoder_save_path)
        logger.info(f"Label encoder saved to {encoder_save_path}")
    except Exception as e:
        logger.error(f"Failed to save label encoder: {e}", exc_info=True)
        return None, None, None  # Cannot proceed without encoder

    return task_embeddings, agent_labels_encoded, num_classes


def build_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    """Builds a simple MLP model for agent classification."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(
                num_classes, activation="softmax"
            ),  # Softmax for multi-class probability
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",  # Use sparse CE for integer labels
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


# --- Main Training Logic ---

if __name__ == "__main__":
    logger.info("--- Starting Agent Selector Model Training ---")

    # 1. Load Data
    data = load_data(LOG_FILE)
    if data is None:
        logger.error("Failed to load data. Exiting.")
        exit()
    if len(data) < BATCH_SIZE * 2:  # Need enough data for at least one train/test batch
        logger.error(
            f"Not enough data to train (found {len(data)} records). Need at least {BATCH_SIZE * 2}. Exiting."
        )
        exit()

    # 2. Load Embedding Model
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}. Exiting.", exc_info=True)
        exit()

    # 3. Preprocess Data
    X, y, n_classes = preprocess_data(data, embedder, ENCODER_SAVE_PATH)
    if X is None or y is None or n_classes is None:
        logger.error("Failed to preprocess data. Exiting.")
        exit()
    if n_classes <= 1:
        logger.error(
            f"Only found {n_classes} agent class(es) in the data. Need at least 2 for classification. Exiting."
        )
        exit()

    # 4. Split Data
    logger.info(f"Splitting data (Test size: {TEST_SPLIT_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Stratify for balanced classes
    )
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 5. Build Model
    input_dim = X_train.shape[1]  # Dimension of embeddings
    model = build_model((input_dim,), n_classes)

    # 6. Train Model
    logger.info(
        f"Starting model training ({EPOCHS} epochs, Batch size: {BATCH_SIZE})..."
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  # Use part of training data for validation during training
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],  # Stop early if no improvement
    )
    logger.info("Model training finished.")

    # 7. Evaluate Model
    logger.info("Evaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # 8. Save Model
    try:
        model.save(MODEL_SAVE_PATH)
        logger.info(f"Trained model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save trained model: {e}", exc_info=True)

    logger.info("--- Agent Selector Model Training Complete ---")
