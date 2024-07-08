import numpy as np

from .kernel import Kernel


class GaussianProcess:
    """
    GaussianProcess class for performing Gaussian Process Regression.

    Attributes:
        kernel (Kernel): Kernel function to compute covariance.
        sigma_n (float): Noise parameter for the Gaussian process.
    """

    def __init__(self, kernel: Kernel, sigma_n: float = 1e-6):
        """
        Initializes the GaussianProcess with a specified kernel and noise parameter.

        Args:
            kernel (Kernel): The kernel function used to compute the covariance matrix.
            sigma_n (float, optional): Observation noise. Defaults to 1e-6.
        """
        self.kernel = kernel
        self.sigma_n = sigma_n

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        """
        Fits the Gaussian Process to the provided training data.

        Args:
            xs (np.ndarray): Training inputs, shape (n_samples, n_features).
            ys (np.ndarray): Training outputs, shape (n_samples,).

        Sets:
            self.xs (np.ndarray): Training inputs.
            self.ys (np.ndarray): Training outputs.
            self.k (np.ndarray): Covariance matrix of the training data.
            self.k_inv (np.ndarray): Inverse of the covariance matrix.
        """
        self.xs = xs
        self.ys = ys
        self.k = self.kernel(xs, xs) + self.sigma_n * np.eye(len(xs))
        self.k_inv = np.linalg.inv(self.k)

    def predict(self, x: np.ndarray):
        """
        Predicts the mean and covariance of the Gaussian process at new input points.

        Args:
            x (np.ndarray): New input points, shape (n_new_samples, n_features).

        Returns:
            tuple: A tuple containing:
                - mu_s (np.ndarray): Predicted means, shape (n_new_samples,).
                - cov_s (np.ndarray): Predicted covariances, shape (n_new_samples, n_new_samples).
        """
        k_s = self.kernel(x, self.xs)
        k_ss = self.kernel(x, x) + self.sigma_n * np.eye(len(x))

        mu_s = k_s.dot(self.k_inv).dot(self.ys)
        cov_s = k_ss - k_s.dot(self.k_inv).dot(k_s.T)

        return mu_s, cov_s
