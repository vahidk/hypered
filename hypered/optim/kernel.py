from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist


class Kernel(ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Registers a new subclass in the kernel registry.

        Args:
            cls: The subclass being initialized.
            **kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        cls._registry[cls.NAME] = cls

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """
        Creates an instance of a kernel subclass.

        Args:
            name (str): The name of the kernel subclass.
            *args: Positional arguments to pass to the kernel subclass.
            **kwargs: Keyword arguments to pass to the kernel subclass.

        Returns:
            An instance of the specified kernel subclass.
        """
        return cls._registry[name](*args, **kwargs)

    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        """
        Computes the kernel between two sets of inputs.

        Args:
            x1 (np.ndarray): The first input array.
            x2 (np.ndarray): The second input array.

        Returns:
            A kernel matrix as a numpy array.
        """
        pass


class RBF(Kernel):
    NAME = "RBF"

    def __init__(self, scale=1.0):
        """
        Initializes the RBF kernel.

        Args:
            scale (float): The length scale parameter of the RBF kernel.
        """
        self.scale = scale

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        """
        Computes the RBF kernel between two sets of inputs.

        Args:
            x1 (np.ndarray): The first input array.
            x2 (np.ndarray): The second input array.

        Returns:
            np.ndarray: The RBF kernel matrix.
        """
        distance = cdist(
            x1 / self.scale,
            x2 / self.scale,
            metric="sqeuclidean",
        )
        return np.exp(-0.5 * distance)


class Matern(Kernel):
    NAME = "Matern"

    def __init__(self, nu: float = 1.5, scale: float = 1.0):
        """
        Initializes the Matern kernel.

        Args:
            nu (float): The smoothness parameter of the Matern kernel.
            scale (float): The length scale parameter of the Matern kernel.
        """
        self.nu = nu
        self.scale = scale

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        """
        Computes the Matern kernel between two sets of inputs.

        Args:
            x1 (np.ndarray): The first input array.
            x2 (np.ndarray): The second input array.

        Returns:
            np.ndarray: The Matern kernel matrix.

        Raises:
            ValueError: If nu is not one of the supported values (0.5, 1.5, 2.5).
        """
        distance = cdist(
            x1 / self.scale,
            x2 / self.scale,
            metric="euclidean",
        )

        if self.nu == 0.5:
            return np.exp(-distance)
        elif self.nu == 1.5:
            return (1 + np.sqrt(3) * distance) * np.exp(-np.sqrt(3) * distance)
        elif self.nu == 2.5:
            return (
                1 + np.sqrt(5) * distance + (5.0 / 3.0) * np.square(distance)
            ) * np.exp(-np.sqrt(5) * distance)
        else:
            raise ValueError("Unsupported value for nu. Use 0.5, 1.5, or 2.5.")
