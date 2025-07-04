import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import os
import sys

# Mock the tensorflow and sklearn modules before importing the tool
# This prevents actual loading of heavy libraries during testing
tf_mock = MagicMock()
metrics_mock = MagicMock()
np_mock = (
    MagicMock()
)  # Mock numpy if needed for specific checks, though usually direct use is fine

# Mock specific functions/classes used by the tool
tf_mock.keras.models.load_model = MagicMock()
tf_mock.keras.preprocessing.image.ImageDataGenerator = MagicMock()

# Mock metrics functions
metrics_mock.accuracy_score = MagicMock(return_value=0.95)
metrics_mock.precision_score = MagicMock(return_value=0.90)
metrics_mock.recall_score = MagicMock(return_value=0.85)
metrics_mock.f1_score = MagicMock(return_value=0.87)
# Return a numpy array for confusion matrix as the real function does
metrics_mock.confusion_matrix = MagicMock(
    return_value=np.array([[50, 5], [10, 35]])
)  # TN=50, FP=5, FN=10, TP=35

# Need to mock os.path.exists and os.path.isdir as well
os_mock = MagicMock()
os_mock.path = MagicMock()
os_mock.path.exists = MagicMock(return_value=True)
os_mock.path.isdir = MagicMock(return_value=True)

# Apply mocks using patch.dict for modules
# Important: Ensure all necessary submodules are mocked if accessed directly
module_mocks = {
    "tensorflow": tf_mock,
    "tensorflow.keras": tf_mock.keras,
    "tensorflow.keras.models": tf_mock.keras.models,
    "tensorflow.keras.preprocessing": tf_mock.keras.preprocessing,
    "tensorflow.keras.preprocessing.image": tf_mock.keras.preprocessing.image,
    "sklearn": MagicMock(),  # Mock base sklearn if needed
    "sklearn.metrics": metrics_mock,
    "numpy": np,  # Use real numpy for array operations in test, but mock if needed
    "os": os_mock,
    "logging": MagicMock(),  # Mock logging to avoid output during tests
}

# Use patch.dict to inject mocks *before* importing the tool
# This ensures the tool sees the mocks when it's loaded
patcher = patch.dict(sys.modules, module_mocks)
patcher.start()

# Now import the tool - it will use the mocked versions
from tools import evaluate_binary_image_classifier


class TestEvaluateBinaryImageClassifier(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test to ensure test isolation
        tf_mock.reset_mock()
        metrics_mock.reset_mock()
        os_mock.reset_mock()
        os_mock.path.exists.return_value = True  # Default to paths existing
        os_mock.path.isdir.return_value = True

        # Configure mock model and generator for a typical scenario
        self.mock_model = MagicMock()
        # Simulate predict output (probabilities) for 100 samples
        # Use a fixed seed for reproducibility if needed, but random is often fine for structure tests
        self.predict_output = np.random.rand(100, 1)
        self.mock_model.predict.return_value = self.predict_output
        tf_mock.keras.models.load_model.return_value = self.mock_model

        self.mock_generator_instance = MagicMock()
        self.mock_test_generator = MagicMock()
        # Simulate 100 samples, with binary classes
        self.mock_test_generator.samples = 100
        # Generate some plausible true classes
        self.true_classes = np.random.randint(0, 2, size=100)
        self.mock_test_generator.classes = self.true_classes
        self.mock_test_generator.class_indices = {"class_0": 0, "class_1": 1}
        # Ensure flow_from_directory returns the configured mock generator
        self.mock_generator_instance.flow_from_directory.return_value = (
            self.mock_test_generator
        )
        # Ensure the ImageDataGenerator constructor returns the instance that has flow_from_directory
        tf_mock.keras.preprocessing.image.ImageDataGenerator.return_value = (
            self.mock_generator_instance
        )

        # Reset mock return values for metrics to defaults
        metrics_mock.accuracy_score.return_value = 0.95
        metrics_mock.precision_score.return_value = 0.90
        metrics_mock.recall_score.return_value = 0.85
        metrics_mock.f1_score.return_value = 0.87
        metrics_mock.confusion_matrix.return_value = np.array([[50, 5], [10, 35]])

    @classmethod
    def tearDownClass(cls):
        # Stop the patcher after all tests in the class have run
        patcher.stop()

    def test_successful_evaluation(self):
        """Test the normal successful execution path"""
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150],
                    "batch_size": 32,
                }
            )
        )

        # --- Assertions ---
        # 1. Check input validation calls
        os_mock.path.exists.assert_called_once_with("/fake/model.h5")
        os_mock.path.isdir.assert_called_once_with("/fake/test_data")

        # 2. Check model loading
        tf_mock.keras.models.load_model.assert_called_once_with("/fake/model.h5")

        # 3. Check ImageDataGenerator setup and flow_from_directory
        tf_mock.keras.preprocessing.image.ImageDataGenerator.assert_called_once_with(
            rescale=1.0 / 255
        )
        self.mock_generator_instance.flow_from_directory.assert_called_once_with(
            "/fake/test_data",
            target_size=(150, 150),  # Ensure tuple conversion
            batch_size=32,
            class_mode="binary",
            shuffle=False,
        )

        # 4. Check model prediction
        # Calculate expected steps: ceil(100 / 32) = 4
        expected_steps = (100 + 32 - 1) // 32
        self.mock_model.predict.assert_called_once_with(
            self.mock_test_generator, steps=expected_steps
        )

        # 5. Check metrics calculations
        # Calculate expected predicted classes based on the mock predict output
        expected_predicted_classes = (self.predict_output > 0.5).astype(int).flatten()

        # Ensure metrics functions were called with the correct true and predicted labels
        metrics_mock.accuracy_score.assert_called_once()
        np.testing.assert_array_equal(
            metrics_mock.accuracy_score.call_args[0][0], self.true_classes
        )
        np.testing.assert_array_equal(
            metrics_mock.accuracy_score.call_args[0][1], expected_predicted_classes
        )

        metrics_mock.precision_score.assert_called_once()
        np.testing.assert_array_equal(
            metrics_mock.precision_score.call_args[0][0], self.true_classes
        )
        np.testing.assert_array_equal(
            metrics_mock.precision_score.call_args[0][1], expected_predicted_classes
        )
        self.assertEqual(
            metrics_mock.precision_score.call_args[1]["zero_division"], 0
        )  # Check kwarg

        metrics_mock.recall_score.assert_called_once()
        np.testing.assert_array_equal(
            metrics_mock.recall_score.call_args[0][0], self.true_classes
        )
        np.testing.assert_array_equal(
            metrics_mock.recall_score.call_args[0][1], expected_predicted_classes
        )
        self.assertEqual(
            metrics_mock.recall_score.call_args[1]["zero_division"], 0
        )  # Check kwarg

        metrics_mock.f1_score.assert_called_once()
        np.testing.assert_array_equal(
            metrics_mock.f1_score.call_args[0][0], self.true_classes
        )
        np.testing.assert_array_equal(
            metrics_mock.f1_score.call_args[0][1], expected_predicted_classes
        )
        self.assertEqual(
            metrics_mock.f1_score.call_args[1]["zero_division"], 0
        )  # Check kwarg

        metrics_mock.confusion_matrix.assert_called_once()
        np.testing.assert_array_equal(
            metrics_mock.confusion_matrix.call_args[0][0], self.true_classes
        )
        np.testing.assert_array_equal(
            metrics_mock.confusion_matrix.call_args[0][1], expected_predicted_classes
        )

        # 6. Check the structure and content of the returned dict
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
        self.assertIn("confusion_matrix", result)
        # Check values returned by mocks
        self.assertEqual(result["accuracy"], 0.95)
        self.assertEqual(result["precision"], 0.90)
        self.assertEqual(result["recall"], 0.85)
        self.assertEqual(result["f1_score"], 0.87)
        # Check list conversion of the numpy array returned by the mock
        self.assertEqual(result["confusion_matrix"], [[50, 5], [10, 35]])

    def test_model_path_not_found(self):
        """Test error handling when model file does not exist"""
        os_mock.path.exists.return_value = False  # Simulate model file not existing
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/nonexistent_model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("Model file not found" in result["error"])
        # Ensure model loading and subsequent steps were not attempted
        tf_mock.keras.models.load_model.assert_not_called()
        self.mock_generator_instance.flow_from_directory.assert_not_called()

    def test_data_dir_not_found(self):
        """Test error handling when test data directory does not exist"""
        os_mock.path.isdir.return_value = False  # Simulate test data dir not existing
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/nonexistent_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("Test data directory not found" in result["error"])
        # Ensure model loading was not attempted (should check dir first)
        tf_mock.keras.models.load_model.assert_not_called()

    def test_invalid_target_size_format(self):
        """Test error handling for incorrect target_size list format"""
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150],  # Invalid size (length != 2)
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("target_size must be a list of two integers" in result["error"])

    def test_invalid_target_size_type(self):
        """Test error handling for non-integer types in target_size"""
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150.5],  # Invalid type (float)
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("target_size must contain only integers" in result["error"])

    def test_no_images_found(self):
        """Test error handling when the generator finds no images"""
        self.mock_test_generator.samples = 0  # Simulate no images found
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/empty_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("No images found" in result["error"])
        # Ensure prediction and metrics were not attempted
        self.mock_model.predict.assert_not_called()
        metrics_mock.accuracy_score.assert_not_called()

    def test_prediction_error(self):
        """Test error handling when model.predict raises an exception"""
        self.mock_model.predict.side_effect = Exception("Prediction failed!")
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue(
            "An unexpected error occurred: Prediction failed!" in result["error"]
        )

    def test_metrics_calculation_error(self):
        """Test error handling when a scikit-learn metric function raises an exception"""
        metrics_mock.accuracy_score.side_effect = Exception(
            "Metrics calculation failed!"
        )
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue(
            "An unexpected error occurred: Metrics calculation failed!"
            in result["error"]
        )

    def test_mismatched_labels_predictions(self):
        """Test error handling when labels and predictions counts don't match"""
        # Simulate predict returning a different number of results than expected samples
        self.mock_model.predict.return_value = np.random.rand(
            99, 1
        )  # 99 predictions instead of 100
        result = (
            evaluate_binary_image_classifier.evaluate_binary_image_classifier.invoke(
                {
                    "saved_model_path": "/fake/model.h5",
                    "test_data_dir": "/fake/test_data",
                    "target_size": [150, 150],
                }
            )
        )
        self.assertIn("error", result)
        self.assertTrue("Mismatch between number of true labels" in result["error"])


if __name__ == "__main__":
    # Need to manage the patcher if running the script directly
    # However, typically tests are run via 'python -m unittest ...'
    # If running directly, ensure mocks are active:
    # with patch.dict(sys.modules, module_mocks):
    #     unittest.main()
    # For standard execution via unittest runner, the class setup/teardown handles it.
    unittest.main()
