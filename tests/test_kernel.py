import unittest
import numpy as np
from scipy.spatial.distance import cdist

from hypered.optim.kernel import RBF, Matern


class TestRBFKernel(unittest.TestCase):
    def test_rbf_kernel(self):
        rbf = RBF(scale=1.0)
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[0, 0], [1, 1]])

        expected_result = np.exp(-0.5 * cdist(x1, x2, metric="sqeuclidean"))
        result = rbf(x1, x2)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_rbf_kernel_different_scale(self):
        rbf = RBF(scale=2.0)
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[0, 0], [1, 1]])

        expected_result = np.exp(-0.5 * cdist(x1 / 2.0, x2 / 2.0, metric="sqeuclidean"))
        result = rbf(x1, x2)
        np.testing.assert_array_almost_equal(result, expected_result)


class TestMaternKernel(unittest.TestCase):
    def test_matern_kernel_nu_0_5(self):
        matern = Matern(nu=0.5, scale=1.0)
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[0, 0], [1, 1]])

        distance = cdist(x1, x2, metric="euclidean")
        expected_result = np.exp(-distance)
        result = matern(x1, x2)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_matern_kernel_nu_1_5(self):
        matern = Matern(nu=1.5, scale=1.0)
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[0, 0], [1, 1]])

        distance = cdist(x1, x2, metric="euclidean")
        expected_result = (1 + np.sqrt(3) * distance) * np.exp(-np.sqrt(3) * distance)
        result = matern(x1, x2)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_matern_kernel_nu_2_5(self):
        matern = Matern(nu=2.5, scale=1.0)
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[0, 0], [1, 1]])

        distance = cdist(x1, x2, metric="euclidean")
        expected_result = (
            1 + np.sqrt(5) * distance + (5.0 / 3.0) * np.square(distance)
        ) * np.exp(-np.sqrt(5) * distance)
        result = matern(x1, x2)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_matern_kernel_invalid_nu(self):
        with self.assertRaises(ValueError):
            matern = Matern(nu=1.0, scale=1.0)
            x1 = np.array([[0, 0], [1, 1]])
            x2 = np.array([[0, 0], [1, 1]])
            matern(x1, x2)


if __name__ == "__main__":
    unittest.main()
