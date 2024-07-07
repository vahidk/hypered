"""Hyperparameter optimizer interface.

This module provides an interface for defining hyperparameter optimization spaces.
It includes functions and classes to create uniform, log-uniform, real, integer,
and categorical search spaces.
"""

from .registry import exportable
from ..optim import space


class variable(exportable):
    """
    Base class for defining different types of variables in hyperparameter optimization.

    This class is automatically registered for export in the registry.
    """

    def __call__(self):
        pass


class real(variable):
    """
    Class for defining a real-valued hyperparameter.

    Args:
        low (float): The lower bound of the search space.
        high (float): The upper bound of the search space.

    Methods:
        __call__(): Creates and returns a `Real` object representing the real-valued search space.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self):
        """
        Creates a `Real` object.

        Returns:
            Real: The real-valued search space object.
        """
        var = space.Real(self.low, self.high)
        return var


class integer(variable):
    """
    Class for defining an integer-valued hyperparameter.

    Args:
        low (int): The lower bound of the search space.
        high (int): The upper bound of the search space.

    Methods:
        __call__(): Creates and returns a `Integer` object representing the integer-valued search space.
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self):
        """
        Creates a `space.Integer` object.

        Returns:
            Integer: The integer-valued search space object.
        """
        var = space.Integer(self.low, self.high)
        return var


class categorical(variable):
    """
    Class for defining a categorical hyperparameter.

    Args:
        categories (list): A list of possible categories for the hyperparameter.

    Methods:
        __call__(): Creates and returns a `Categorical` object representing the categorical search space.
    """

    def __init__(self, categories: list):
        self.categories = categories

    def __call__(self):
        """
        Creates a `Categorical` object.

        Returns:
            Categorical: The categorical search space object.
        """
        var = space.Categorical(self.categories)
        return var
