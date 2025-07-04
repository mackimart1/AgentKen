import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    MobileNetV2,
    VGG16,
    ResNet50,
)  # Add others as needed
import uuid
import os
from langchain_core.tools import tool
import traceback

# Mapping base model names to their Keras application classes and preprocess functions
BASE_MODEL_MAP = {
    "MobileNetV2": (MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "VGG16": (VGG16, tf.keras.applications.vgg16.preprocess_input),
    "ResNet50": (ResNet50, tf.keras.applications.resnet50.preprocess_input),
    # Add more models here if needed
}


@tool
def train_image_classifier(
    train_data_dir: str,  # Changed type hint from DirectoryIterator to str (path)
    validation_data_dir: str,  # Changed type hint from DirectoryIterator to str (path)
    base_model_name: str = "MobileNetV2",
    image_size: list[int] = [224, 224],  # Changed type hint from tuple to list[int]
    learning_rate: float = 0.001,
    epochs: int = 5,
) -> dict:
    """
    Trains an image classification model using transfer learning.

    Args:
        train_data_dir: Path to the directory containing the training dataset.
        validation_data_dir: Path to the directory containing the validation dataset.
        base_model_name: Name of the pre-trained base model (e.g., "MobileNetV2", "VGG16", "ResNet50").
        image_size: The target size of the images as a list [height, width].
        learning_rate: The learning rate for the Adam optimizer.
        epochs: The number of epochs to train for.

    Returns:
        A dictionary containing:
        - 'saved_model_path': The file path where the trained model is saved (H5 format).
        - 'training_history': A dictionary containing training history (loss, accuracy, val_loss, val_accuracy).
        - 'error': An error message if training failed, otherwise None.
    """
    # --- Internal Data Loading (Needs Implementation based on paths) ---
    # TODO: Implement logic here to create train_generator and validation_generator
    #       from train_data_dir and validation_data_dir using tf.keras.preprocessing.image.ImageDataGenerator
    # Example (needs refinement based on actual data structure and preprocessing needs):
    # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(...)
    # validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(...)
    # train_generator = train_datagen.flow_from_directory(train_data_dir, ...)
    # validation_generator = validation_datagen.flow_from_directory(validation_data_dir, ...)
    # --- End Internal Data Loading ---

    # Placeholder check until internal loading is implemented
    # This check will always fail now as strings don't have 'num_classes'
    # Remove or adapt this check once internal loading is done.
    # if not hasattr(train_generator, 'num_classes') or not hasattr(validation_generator, 'num_classes'):
    #      return {"saved_model_path": None, "training_history": None, "error": "Internal data loading not yet implemented for train/validation directories."}

    # --- TEMPORARY EARLY EXIT (Remove after implementing internal loading) ---
    return {
        "saved_model_path": None,
        "training_history": None,
        "error": "Tool signature fixed, but internal data loading from paths needs implementation.",
    }
    # --- END TEMPORARY EARLY EXIT ---

    try:  # Keep the rest of the try block, but it won't be reached yet
        if base_model_name not in BASE_MODEL_MAP:
            return {
                "saved_model_path": None,
                "training_history": None,
                "error": f"Base model '{base_model_name}' not supported.",
            }

        BaseCtor, _ = BASE_MODEL_MAP[base_model_name]
        # Convert image_size list back to tuple for Keras
        img_shape = tuple(image_size) + (3,)  # Add channel dimension

        # Load base model
        base_model = BaseCtor(
            input_shape=img_shape, include_top=False, weights="imagenet"
        )
        base_model.trainable = False  # Freeze weights

        # Create the new model
        inputs = tf.keras.Input(shape=img_shape)
        # Note: Preprocessing is usually done *before* the generator,
        # but if not, it should be added as a Lambda layer here.
        # Assuming generators handle preprocessing.
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)  # Add dropout for regularization

        # Infer number of classes from generator
        num_classes = train_generator.num_classes
        if num_classes == 0:
            return {
                "saved_model_path": None,
                "training_history": None,
                "error": "Train generator reported 0 classes.",
            }
        elif num_classes == 1:
            # Ambiguous case, could be binary (0/1) or single class regression. Assume binary based on prompt.
            activation = "sigmoid"
            loss = "binary_crossentropy"
            output_units = 1
        elif num_classes == 2:
            # Often treated as binary classification with 1 output unit + sigmoid
            activation = "sigmoid"
            loss = "binary_crossentropy"
            output_units = 1
        else:  # Multiclass
            activation = "softmax"
            loss = "categorical_crossentropy"
            output_units = num_classes

        outputs = layers.Dense(output_units, activation=activation)(x)
        model = models.Model(inputs, outputs)

        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )

        # Train the model
        # NOTE: Passing actual generators like this won't work if the tool is called remotely.
        # This code assumes local execution context where generators are valid.
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            # Add steps_per_epoch and validation_steps if generators are infinite
            # steps_per_epoch=train_generator.samples // train_generator.batch_size,
            # validation_steps=validation_generator.samples // validation_generator.batch_size,
        )

        # Save the model
        model_dir = "trained_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Use a unique filename
        model_filename = f"{base_model_name}_{uuid.uuid4()}.h5"
        saved_model_path = os.path.join(model_dir, model_filename)
        model.save(saved_model_path)

        # Convert history to standard dict
        history_dict = history.history

        return {
            "saved_model_path": saved_model_path,
            "training_history": history_dict,
            "error": None,
        }

    except Exception as e:
        error_message = (
            f"An error occurred during training: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_message)  # Print for debugging logs
        return {
            "saved_model_path": None,
            "training_history": None,
            "error": f"An error occurred during training: {str(e)}",
        }
