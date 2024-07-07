from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm


class AcquisitionFn(ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register the subclass in the acquisition function registry.

        Parameters:
        cls (class): The subclass being initialized.
        kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        cls._registry[cls.NAME] = cls

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create an instance of a registered acquisition function.

        Parameters:
        name (str): The name of the acquisition function to create.
        args: Additional positional arguments to pass to the acquisition function's initializer.
        kwargs: Additional keyword arguments to pass to the acquisition function's initializer.

        Returns:
        AcquisitionFn: An instance of the specified acquisition function.
        """
        return cls._registry[name](*args, **kwargs)

    @abstractmethod
    def __call__(self, model, x, y_opt):
        """
        Evaluate the acquisition function.

        Parameters:
        model: The surrogate model used for prediction.
        x (ndarray): The input data for which the acquisition function is evaluated.
        y_opt (float): The current best observed value.

        Returns:
        ndarray: The evaluated acquisition function values.
        """
        pass


class UpperConfidenceBound(AcquisitionFn):
    NAME = "UCB"

    def __init__(self, kappa=2.576):
        """
        Initialize the Upper Confidence Bound acquisition function.

        Parameters:
        kappa (float): The exploration-exploitation trade-off parameter.
        """
        self.kappa = kappa

    def __call__(self, model, x, y_opt):
        """
        Evaluate the Upper Confidence Bound acquisition function.

        Parameters:
        model: The surrogate model used for prediction.
        x (ndarray): The input data for which the acquisition function is evaluated.
        y_opt (float): The current best observed value.

        Returns:
        ndarray: The evaluated Upper Confidence Bound values.
        """
        mu, cov = model.predict(x)
        sigma = np.sqrt(np.diag(cov))
        return mu + self.kappa * sigma


class ExpectedImprovement(AcquisitionFn):
    NAME = "EI"

    def __init__(self, xi=0.01):
        """
        Initialize the Expected Improvement acquisition function.

        Parameters:
        xi (float): The exploration-exploitation trade-off parameter.
        """
        self.xi = xi

    def __call__(self, model, x, y_opt):
        """
        Evaluate the Expected Improvement acquisition function.

        Parameters:
        model: The surrogate model used for prediction.
        x (ndarray): The input data for which the acquisition function is evaluated.
        y_opt (float): The current best observed value.

        Returns:
        ndarray: The evaluated Expected Improvement values.
        """
        mu, cov = model.predict(x)
        sigma = np.sqrt(np.diag(cov))
        imp = mu - y_opt - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
