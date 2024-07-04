"""Hyperparameter optimizer interface.

This module provides an interface for defining hyperparameter optimization spaces using the `skopt` library. 
It includes functions and classes to create uniform, log-uniform, real, integer, and categorical search spaces.
"""

import skopt

from .common import registry


@registry.export
def uniform():
    """
    Returns the string identifier for a uniform distribution.

    Returns:
        str: The string "uniform".
    """
    return "uniform"


@registry.export
def log_uniform():
    """
    Returns the string identifier for a log-uniform distribution.

    Returns:
        str: The string "log-uniform".
    """
    return "log-uniform"


class variable(registry.exportable):
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
        prior (callable, optional): The prior distribution to use, either uniform or log-uniform. Defaults to uniform.

    Methods:
        __call__(): Creates and returns a `skopt.space.Real` object representing the real-valued search space.
    """

    def __init__(self, low, high, prior=uniform):
        self.low = low
        self.high = high
        self.prior = prior

    def __call__(self):
        """
        Creates a `skopt.space.Real` object.

        Returns:
            skopt.space.Real: The real-valued search space object.
        """
        var = skopt.space.Real(self.low, self.high, self.prior())
        return var


class integer(variable):
    """
    Class for defining an integer-valued hyperparameter.

    Args:
        low (int): The lower bound of the search space.
        high (int): The upper bound of the search space.

    Methods:
        __call__(): Creates and returns a `skopt.space.Integer` object representing the integer-valued search space.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self):
        """
        Creates a `skopt.space.Integer` object.

        Returns:
            skopt.space.Integer: The integer-valued search space object.
        """
        var = skopt.space.Integer(self.low, self.high)
        return var


class categorical(variable):
    """
    Class for defining a categorical hyperparameter.

    Args:
        categories (list): A list of possible categories for the hyperparameter.

    Methods:
        __call__(): Creates and returns a `skopt.space.Categorical` object representing the categorical search space.
    """

    def __init__(self, categories):
        self.categories = categories

    def __call__(self):
        """
        Creates a `skopt.space.Categorical` object.

        Returns:
            skopt.space.Categorical: The categorical search space object.
        """
        var = skopt.space.Categorical(self.categories)
        return var
