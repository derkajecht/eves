import unittest
import numpy as np
import os

from src.micro_math import relu, relu_derivative, softmax, ce_loss, one_hot

class TestNeuralNetworkUtils(unittest.TestCase):

    def test_relu(self):
        z = np.array([-1, 0, 2])
        expected = np.array([0, 0, 2])
        np.testing.assert_array_equal(relu(z), expected)

    def test_relu_derivative(self):
        z = np.array([-1, 0, 2])
        # Note: derivative at 0 is technically undefined, but usually 0 in code
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(z), expected)

    def test_softmax(self):
        z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        probs = softmax(z)

        # Check if rows sum to 1
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, np.array([1.0, 1.0]))

        # Check if values are within [0, 1]
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))

    def test_ce_loss(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.8, 0.1])
        # Manual calculation: -ln(0.8) approx 0.2231
        loss = ce_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.2231435513)

    def test_one_hot(self):
        # Create a temporary .npy file to test the loader
        test_file = "temp_labels.npy"
        labels = np.array([0, 2, 1])
        np.save(test_file, labels)

        try:
            oh = one_hot(test_file, num_classes=3)
            expected = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ])
            np.testing.assert_array_equal(oh, expected)
        finally:
            # Clean up the file after testing
            if os.path.exists(test_file):
                os.remove(test_file)

if __name__ == "__main__":
    unittest.main()
