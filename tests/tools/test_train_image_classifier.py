import unittest
from unittest.mock import patch, MagicMock, ANY, PropertyMock
import os
import shutil
import uuid
import sys
import importlib

# --- Mocking Setup ---
# Mock TensorFlow and Keras *before* importing the tool
# This prevents actual TF loading during testing and avoids heavy dependency issues
mock_tf = MagicMock(name="tensorflow")
mock_keras = MagicMock(name="keras")
mock_layers = MagicMock(name="keras.layers")
mock_models = MagicMock(name="keras.models")
mock_optimizers = MagicMock(name="keras.optimizers")
mock_applications = MagicMock(name="keras.applications")
mock_preprocessing = MagicMock(name="keras.preprocessing")
mock_image = MagicMock(name="keras.preprocessing.image")

# Place mocks in sys.modules
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.layers"] = mock_layers
sys.modules["tensorflow.keras.models"] = mock_models
sys.modules["tensorflow.keras.optimizers"] = mock_optimizers
sys.modules["tensorflow.keras.applications"] = mock_applications
sys.modules["tensorflow.keras.preprocessing"] = mock_preprocessing
sys.modules["tensorflow.keras.preprocessing.image"] = mock_image

# Mock specific components needed by the tool
mock_base_model_instance = MagicMock(name="BaseModelInstance")
mock_model_instance = MagicMock(name="ModelInstance")
mock_history = MagicMock(name="History")
mock_history.history = {
    "loss": [0.5],
    "accuracy": [0.8],
    "val_loss": [0.6],
    "val_accuracy": [0.75],
}
mock_model_instance.fit.return_value = mock_history
mock_model_instance.save.return_value = None

# Mock Keras Application classes and their preprocess functions
mock_MobileNetV2 = MagicMock(
    name="MobileNetV2_Class", return_value=mock_base_model_instance
)
mock_VGG16 = MagicMock(name="VGG16_Class", return_value=mock_base_model_instance)
mock_ResNet50 = MagicMock(name="ResNet50_Class", return_value=mock_base_model_instance)
mock_applications.MobileNetV2 = mock_MobileNetV2
mock_applications.VGG16 = mock_VGG16
mock_applications.ResNet50 = mock_ResNet50
# Mock the submodules for preprocess_input if needed (though not directly used in this tool's logic)
mock_applications.mobilenet_v2 = MagicMock()
mock_applications.vgg16 = MagicMock()
mock_applications.resnet50 = MagicMock()
mock_applications.mobilenet_v2.preprocess_input = MagicMock(
    name="MobileNetV2_Preprocess"
)
mock_applications.vgg16.preprocess_input = MagicMock(name="VGG16_Preprocess")
mock_applications.resnet50.preprocess_input = MagicMock(name="ResNet50_Preprocess")

# Mock other Keras components
mock_Input = MagicMock(name="InputLayer")
mock_layers.Input = mock_Input
mock_layers.GlobalAveragePooling2D.return_value = MagicMock(name="GAP2DInstance")
mock_layers.Dropout.return_value = MagicMock(name="DropoutInstance")
mock_layers.Dense.return_value = MagicMock(name="DenseInstance")
mock_models.Model.return_value = mock_model_instance
mock_optimizers.Adam.return_value = MagicMock(name="AdamInstance")

# Mock DirectoryIterator class and instances
mock_generator_class = MagicMock(name="DirectoryIteratorClass")
mock_image.DirectoryIterator = mock_generator_class
mock_train_generator_instance = MagicMock(name="TrainGeneratorInstance")
mock_val_generator_instance = MagicMock(name="ValGeneratorInstance")
# Use PropertyMock for attributes like num_classes that the tool reads
type(mock_train_generator_instance).num_classes = PropertyMock(return_value=2)
type(mock_val_generator_instance).num_classes = PropertyMock(return_value=2)
# --- End Mocking Setup ---


# Now import the tool *after* all mocks are set up
# Use importlib.import_module to get the module object
# Ensure the path is correct relative to the execution directory
try:
    train_image_classifier_module = importlib.import_module(
        "tools.train_image_classifier"
    )
except ModuleNotFoundError:
    # If running tests from a different directory structure, adjust path
    # This might happen if tests are run from the root vs. inside the tests dir
    # For simplicity, assume standard structure first. Add error handling if needed.
    print(
        "Error: Could not import tools.train_image_classifier. Check PYTHONPATH or test execution directory."
    )
    sys.exit(1)  # Exit if import fails, as tests cannot run

# Reload the module to ensure it uses the mocked imports defined above
importlib.reload(train_image_classifier_module)
# Get the actual tool function from the reloaded module
train_image_classifier_tool = train_image_classifier_module.train_image_classifier


class TestTrainImageClassifier(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test to ensure test isolation
        mock_tf.reset_mock()
        mock_keras.reset_mock()
        mock_layers.reset_mock()
        mock_models.reset_mock()
        mock_optimizers.reset_mock()
        mock_applications.reset_mock()
        mock_base_model_instance.reset_mock()
        mock_model_instance.reset_mock()
        mock_MobileNetV2.reset_mock()
        mock_VGG16.reset_mock()
        mock_ResNet50.reset_mock()
        mock_train_generator_instance.reset_mock()
        mock_val_generator_instance.reset_mock()
        # Reset num_classes property mocks
        type(mock_train_generator_instance).num_classes = PropertyMock(return_value=2)
        type(mock_val_generator_instance).num_classes = PropertyMock(return_value=2)
        # Reset fit mock specifically as it's patched per-test sometimes
        mock_model_instance.fit.reset_mock(side_effect=None)
        mock_model_instance.fit.return_value = mock_history

        # Ensure the target directory for saving models exists and is empty
        self.model_dir = "trained_models"
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        # We don't create it here; the tool should create it if needed.

    def tearDown(self):
        # Clean up created directories after each test
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

    @patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"))
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_train_image_classifier_success_binary(
        self, mock_exists, mock_makedirs, mock_uuid
    ):
        # Arrange
        # Simulate directory not existing initially, then existing after creation
        mock_exists.side_effect = lambda path: False if path == self.model_dir else True
        type(mock_train_generator_instance).num_classes = PropertyMock(
            return_value=2
        )  # Binary case

        # Act
        result = train_image_classifier_tool.invoke(
            {
                "train_generator": mock_train_generator_instance,
                "validation_generator": mock_val_generator_instance,
                "base_model_name": "MobileNetV2",
                "image_size": (224, 224),
                "learning_rate": 0.001,
                "epochs": 3,
            }
        )

        # Assert
        # Check directory creation
        mock_exists.assert_any_call(
            self.model_dir
        )  # Check if exists was called for the dir
        mock_makedirs.assert_called_once_with(self.model_dir)

        # Check base model loading
        mock_MobileNetV2.assert_called_once_with(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        self.assertEqual(mock_base_model_instance.trainable, False)

        # Check model compilation (binary case)
        mock_model_instance.compile.assert_called_once_with(
            optimizer=ANY,  # Check optimizer type below
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        # Check optimizer instance
        mock_optimizers.Adam.assert_called_once_with(learning_rate=0.001)

        # Check model training
        mock_model_instance.fit.assert_called_once_with(
            mock_train_generator_instance,
            epochs=3,
            validation_data=mock_val_generator_instance,
        )

        # Check model saving
        expected_path = os.path.join(
            self.model_dir, "MobileNetV2_12345678-1234-5678-1234-567812345678.h5"
        )
        mock_model_instance.save.assert_called_once_with(expected_path)

        # Check result structure and content
        self.assertIsNone(
            result.get("error"), f"Expected no error, but got: {result.get('error')}"
        )
        self.assertEqual(result.get("saved_model_path"), expected_path)
        self.assertEqual(result.get("training_history"), mock_history.history)

    @patch("uuid.uuid4", return_value=uuid.UUID("abcdef12-abcd-ef12-abcd-ef12abcdef12"))
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_train_image_classifier_success_multiclass(
        self, mock_exists, mock_makedirs, mock_uuid
    ):
        # Arrange
        mock_exists.side_effect = lambda path: False if path == self.model_dir else True
        type(mock_train_generator_instance).num_classes = PropertyMock(
            return_value=10
        )  # Multiclass

        # Act
        result = train_image_classifier_tool.invoke(
            {
                "train_generator": mock_train_generator_instance,
                "validation_generator": mock_val_generator_instance,
                "base_model_name": "ResNet50",  # Different model
                "image_size": (128, 128),  # Different size
                "learning_rate": 0.01,  # Different LR
                "epochs": 1,  # Different epochs
            }
        )

        # Assert
        mock_exists.assert_any_call(self.model_dir)
        mock_makedirs.assert_called_once_with(self.model_dir)
        mock_ResNet50.assert_called_once_with(
            input_shape=(128, 128, 3), include_top=False, weights="imagenet"
        )
        self.assertEqual(mock_base_model_instance.trainable, False)
        # Check model compilation (multiclass case)
        mock_model_instance.compile.assert_called_once_with(
            optimizer=ANY,
            loss="categorical_crossentropy",  # Check for categorical
            metrics=["accuracy"],
        )
        mock_optimizers.Adam.assert_called_once_with(learning_rate=0.01)
        mock_model_instance.fit.assert_called_once_with(
            mock_train_generator_instance,
            epochs=1,
            validation_data=mock_val_generator_instance,
        )
        expected_path = os.path.join(
            self.model_dir, "ResNet50_abcdef12-abcd-ef12-abcd-ef12abcdef12.h5"
        )
        mock_model_instance.save.assert_called_once_with(expected_path)
        self.assertIsNone(
            result.get("error"), f"Expected no error, but got: {result.get('error')}"
        )
        self.assertEqual(result.get("saved_model_path"), expected_path)
        self.assertEqual(result.get("training_history"), mock_history.history)

    def test_unsupported_base_model(self):
        # Arrange
        type(mock_train_generator_instance).num_classes = PropertyMock(return_value=2)

        # Act
        result = train_image_classifier_tool.invoke(
            {
                "train_generator": mock_train_generator_instance,
                "validation_generator": mock_val_generator_instance,
                "base_model_name": "UnsupportedNet",  # Use an unsupported name
                "image_size": (224, 224),
                "learning_rate": 0.001,
                "epochs": 3,
            }
        )

        # Assert
        self.assertIsNone(result.get("saved_model_path"))
        self.assertIsNone(result.get("training_history"))
        self.assertEqual(
            result.get("error"), "Base model 'UnsupportedNet' not supported."
        )
        mock_model_instance.compile.assert_not_called()
        mock_model_instance.fit.assert_not_called()
        mock_model_instance.save.assert_not_called()

    # Patch the 'fit' method on the *mocked* Model instance returned by the mocked Model constructor
    # This ensures we patch the correct object within the mocked environment
    @patch.object(
        mock_model_instance, "fit", side_effect=Exception("Test Keras Fit Error")
    )
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_keras_exception_handling(
        self, mock_exists, mock_makedirs, mock_fit_error_method
    ):
        # Arrange
        mock_exists.side_effect = lambda path: False if path == self.model_dir else True
        type(mock_train_generator_instance).num_classes = PropertyMock(return_value=2)

        # Act
        result = train_image_classifier_tool.invoke(
            {
                "train_generator": mock_train_generator_instance,
                "validation_generator": mock_val_generator_instance,
                "base_model_name": "MobileNetV2",
                "image_size": (224, 224),
                "learning_rate": 0.001,
                "epochs": 3,
            }
        )

        # Assert
        self.assertIsNone(result.get("saved_model_path"))
        self.assertIsNone(result.get("training_history"))
        # Check that the error message contains the specific exception text
        self.assertIn(
            "An error occurred during training: Test Keras Fit Error",
            result.get("error"),
        )
        # Ensure fit was called (which triggered the exception)
        mock_model_instance.fit.assert_called_once()
        # Ensure save was NOT called after the error
        mock_model_instance.save.assert_not_called()

    def test_invalid_generator_input(self):
        """Tests the initial check for valid generator-like objects."""
        # Arrange
        invalid_generator = (
            "not a generator"  # Pass a string instead of a mocked object
        )

        # Act
        result = train_image_classifier_tool.invoke(
            {
                "train_generator": invalid_generator,
                "validation_generator": mock_val_generator_instance,  # One valid, one invalid
                "base_model_name": "MobileNetV2",
                "image_size": (224, 224),
                "learning_rate": 0.001,
                "epochs": 3,
            }
        )

        # Assert
        self.assertIsNone(result.get("saved_model_path"))
        self.assertIsNone(result.get("training_history"))
        self.assertEqual(
            result.get("error"),
            "Invalid generator objects provided. Data generators cannot be passed directly as arguments in this environment.",
        )
        # Ensure model building/training steps were not reached
        mock_MobileNetV2.assert_not_called()
        mock_model_instance.compile.assert_not_called()
        mock_model_instance.fit.assert_not_called()


if __name__ == "__main__":
    # Ensure the test runner uses the mocked environment
    unittest.main()
