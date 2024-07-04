"""Hyperparameter optimizer interface.

This module provides functions to create minimization and maximization objectives for hyperparameter optimization.
These functions use a utility function to lookup specific metrics from a nested dictionary structure.
"""

from .common import registry, dict_utils


@registry.export
def minimize(name):
    """
    Creates a function to minimize a specified metric.

    Args:
        name (str): The name of the metric to minimize.

    Returns:
        function: A function that takes a dictionary of metrics and returns the value of the specified metric.
    """

    def _func(metrics):
        return dict_utils.lookup_flat(metrics, name)

    return _func


@registry.export
def maximize(name):
    """
    Creates a function to maximize a specified metric.

    Args:
        name (str): The name of the metric to maximize.

    Returns:
        function: A function that takes a dictionary of metrics and returns the negative value of the specified metric.
    """

    def _func(metrics):
        return -dict_utils.lookup_flat(metrics, name)

    return _func
