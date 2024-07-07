"""Hyperparameter optimizer interface.

This module provides functionality for managing output directories and contextual paths 
for experiments, parameters, and results. It also includes a class for managing device IDs 
in a round-robin fashion.
"""

from .registry import exportable

OUTPUT_DIR = "experiments"


class experiment_dir(exportable):
    """
    Class for retrieving the experiment directory from the context.

    Methods:
        __call__(ctx): Returns the experiment directory from the provided context.
    """

    def __call__(self, ctx) -> str:
        """
        Retrieve the experiment directory from the context.

        Args:
            ctx (dict): The context dictionary containing the experiment directory path.

        Returns:
            str: The path to the experiment directory.
        """
        return ctx["experiment_dir"]


class params_path(exportable):
    """
    Class for retrieving the parameters path from the context.

    Methods:
        __call__(ctx): Returns the parameters path from the provided context.
    """

    def __call__(self, ctx) -> str:
        """
        Retrieve the parameters path from the context.

        Args:
            ctx (dict): The context dictionary containing the parameters path.

        Returns:
            str: The path to the parameters file.
        """
        return ctx["params_path"]


class results_path(exportable):
    """
    Class for retrieving the results path from the context.

    Methods:
        __call__(ctx): Returns the results path from the provided context.
    """

    def __call__(self, ctx) -> str:
        """
        Retrieve the results path from the context.

        Args:
            ctx (dict): The context dictionary containing the results path.

        Returns:
            str: The path to the results file.
        """
        return ctx["results_path"]


class device_id(exportable):
    """
    Class for managing device IDs in a round-robin fashion.

    Args:
        count (int): The number of devices available.

    Methods:
        __call__(ctx): Returns the next device ID in a round-robin sequence.
    """

    def __init__(self, count: int):
        self.count = count
        self.index = -1

    def __call__(self, ctx) -> int:
        """
        Retrieve the next device ID in a round-robin sequence.

        Args:
            ctx (dict): The context dictionary (unused in this method).

        Returns:
            int: The next device ID.
        """
        self.index = (self.index + 1) % self.count
        return self.index
