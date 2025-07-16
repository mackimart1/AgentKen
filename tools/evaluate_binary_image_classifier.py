import os
from langchain_core.tools import tool
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def evaluate_binary_image_classifier(
    saved_model_path: str,
    test_data_dir: str,
    target_size: list[int],
    batch_size: int = 32,
) -> dict:
    """
    Evaluates a saved binary image classification model.

    Args:
        saved_model_path: Path to the saved Keras model (.h5 or SavedModel format).
        test_data_dir: Path to the directory containing the test dataset.
                       It should contain subdirectories for each class (e.g., class_0, class_1).
        target_size: A list containing the target height and width for resizing images, e.g., [150, 150].
        batch_size: Batch size for the test generator. Defaults to 32.

    Returns:
        A dictionary containing evaluation metrics:
        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1_score": float,
            "confusion_matrix": list[list[int]]  # [[TN, FP], [FN, TP]]
        }
        Returns an error dictionary if evaluation fails.
    """
    try:
        if not os.path.exists(saved_model_path):
            return {"error": f"Model file not found: {saved_model_path}"}
        if not os.path.isdir(test_data_dir):
            return {"error": f"Test data directory not found: {test_data_dir}"}
        if not isinstance(target_size, list) or len(target_size) != 2:
            return {
                "error": "target_size must be a list of two integers [height, width]"
            }
        if not all(isinstance(x, int) for x in target_size):
            return {"error": "target_size must contain only integers"}

        logger.info(f"Loading model from: {saved_model_path}")
        model = tf.keras.models.load_model(saved_model_path)

        logger.info(f"Setting up test data generator for directory: {test_data_dir}")
        # Only rescale, no augmentation for testing
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        )

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=tuple(target_size),
            batch_size=batch_size,
            class_mode="binary",  # Crucial for binary classification
            shuffle=False,  # Important for matching predictions with labels
        )

        num_test_samples = test_generator.samples
        if num_test_samples == 0:
            return {"error": f"No images found in {test_data_dir}"}

        logger.info(f"Found {num_test_samples} test samples.")
        logger.info(f"Class indices: {test_generator.class_indices}")

        # Get predictions
        logger.info("Generating predictions...")
        # Ensure steps covers all samples, even if not a perfect multiple of batch_size
        steps = (num_test_samples + batch_size - 1) // batch_size
        predictions = model.predict(test_generator, steps=steps)
        # Slice predictions in case steps*batch_size > num_test_samples
        predictions = predictions[:num_test_samples]

        # Convert probabilities to binary predictions (0 or 1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()

        # Get true labels
        true_classes = test_generator.classes
        logger.info(f"True classes shape: {true_classes.shape}")
        logger.info(f"Predicted classes shape: {predicted_classes.shape}")

        if len(true_classes) != len(predicted_classes):
            return {
                "error": f"Mismatch between number of true labels ({len(true_classes)}) and predictions ({len(predicted_classes)}). Check generator steps or data."
            }

        # Calculate metrics
        logger.info("Calculating metrics...")
        accuracy = accuracy_score(true_classes, predicted_classes)
        # Use zero_division=0 to return 0 instead of raising an error for undefined metrics (e.g., precision when TP+FP=0)
        precision = precision_score(true_classes, predicted_classes, zero_division=0)
        recall = recall_score(true_classes, predicted_classes, zero_division=0)
        f1 = f1_score(true_classes, predicted_classes, zero_division=0)
        cm = confusion_matrix(true_classes, predicted_classes)

        # Ensure confusion matrix is 2x2 for binary classification, even if one class is missing in predictions/labels
        if cm.shape != (2, 2):
            # If only one class present, confusion_matrix might return a single number or smaller array.
            # Reconstruct a 2x2 matrix based on which class was present.
            logger.warning(f"Confusion matrix is not 2x2 ({cm.shape}). Reconstructing.")
            if (
                len(np.unique(true_classes)) == 1
                or len(np.unique(predicted_classes)) == 1
            ):
                cm_full = np.zeros((2, 2), dtype=int)
                present_class = np.unique(true_classes)[
                    0
                ]  # Assume true classes dictate the present class
                if present_class == 0:  # Only class 0 present
                    cm_full[0, 0] = (
                        cm[0, 0] if cm.shape == (1, 1) else cm[0] if cm.ndim == 1 else 0
                    )  # TN
                else:  # Only class 1 present
                    cm_full[1, 1] = (
                        cm[0, 0] if cm.shape == (1, 1) else cm[0] if cm.ndim == 1 else 0
                    )  # TP
                cm = cm_full
            else:
                # Fallback for unexpected shapes, though less likely in binary case
                logger.error(
                    f"Unexpected confusion matrix shape: {cm.shape}. Returning as is."
                )

        # Convert confusion matrix numpy array to list of lists for JSON serialization
        cm_list = cm.tolist()

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm_list,  # [[TN, FP], [FN, TP]]
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}
