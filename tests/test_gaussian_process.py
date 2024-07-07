import unittest
import numpy as np
from unittest.mock import MagicMock, call

from hypered.optim.kernel import Kernel
from hypered.optim.gaussian_process import GaussianProcess


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        # Create a mock kernel
        self.kernel = MagicMock(spec=Kernel)
        self.kernel.return_value = np.array([[1, 0.5], [0.5, 1]])

        # Create an instance of GaussianProcess
        self.gp = GaussianProcess(kernel=self.kernel, sigma_n=1e-6)

        # Training data
        self.xs = np.array([[1, 2], [3, 4]])
        self.ys = np.array([5, 6])

    def test_fit(self):
        self.gp.fit(self.xs, self.ys)

        # Check that training data is stored correctly
        np.testing.assert_array_equal(self.gp.xs, self.xs)
        np.testing.assert_array_equal(self.gp.ys, self.ys)

        # Check that the kernel function is called correctly
        self.kernel.assert_called_with(self.xs, self.xs)

        # Check the covariance matrix and its inverse
        expected_k = np.array([[1, 0.5], [0.5, 1]]) + 1e-6 * np.eye(2)
        np.testing.assert_array_almost_equal(self.gp.k, expected_k)
        np.testing.assert_array_almost_equal(self.gp.k_inv, np.linalg.inv(expected_k))

    def test_predict(self):
        self.gp.fit(self.xs, self.ys)

        # New input data
        x_new = np.array([[5, 6], [7, 8]])

        # Set up kernel return values for the new inputs
        self.kernel.side_effect = [
            np.array([[1, 0.1], [0.1, 1]]),  # kernel(x_new, self.xs)
            np.array([[1, 0.2], [0.2, 1]]),  # kernel(x_new, x_new)
        ]

        # Perform prediction
        mu_s, cov_s = self.gp.predict(x_new)

        # Check the predicted mean and covariance
        k_s = np.array([[1, 0.1], [0.1, 1]])
        k_ss = np.array([[1, 0.2], [0.2, 1]]) + 1e-6 * np.eye(2)

        expected_mu_s = k_s.dot(self.gp.k_inv).dot(self.ys)
        expected_cov_s = k_ss - k_s.dot(self.gp.k_inv).dot(k_s.T)

        np.testing.assert_array_almost_equal(mu_s, expected_mu_s)
        np.testing.assert_array_almost_equal(cov_s, expected_cov_s)


if __name__ == "__main__":
    unittest.main()
