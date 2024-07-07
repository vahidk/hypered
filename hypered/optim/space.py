from abc import ABC, abstractmethod
import numpy as np


class Variable(ABC):
    def __init__(self):
        """
        Initialize a Variable instance with a default size of 1.
        """
        self.size = 1

    @abstractmethod
    def denormalize(self, x: np.ndarray):
        """
        Denormalize the given normalized value.

        Parameters:
        x (np.ndarray): Normalized value to be denormalized.

        Returns:
        Denormalized value.
        """
        pass


class Real(Variable):
    def __init__(self, low: float, high: float):
        """
        Initialize a Real variable with the specified range.

        Parameters:
        low (float): The lower bound of the real variable.
        high (float): The upper bound of the real variable.
        """
        super().__init__()
        self.low = low
        self.high = high

    def denormalize(self, x: np.ndarray):
        """
        Denormalize the given normalized value for a real variable.

        Parameters:
        x (np.ndarray): Normalized value to be denormalized.

        Returns:
        float: Denormalized real value.
        """
        return x.item() * (self.high - self.low) + self.low


class Integer(Variable):
    def __init__(self, low: int, high: int):
        """
        Initialize an Integer variable with the specified range.

        Parameters:
        low (int): The lower bound of the integer variable.
        high (int): The upper bound of the integer variable.
        """
        super().__init__()
        self.low = low
        self.high = high

    def denormalize(self, x: np.ndarray):
        """
        Denormalize the given normalized value for an integer variable.

        Parameters:
        x (np.ndarray): Normalized value to be denormalized.

        Returns:
        int: Denormalized integer value.
        """
        return int(x.item() * (self.high - self.low) + self.low)


class Categorical(Variable):
    def __init__(self, categories: list):
        """
        Initialize a Categorical variable with the specified categories.

        Parameters:
        categories (list): List of categories for the categorical variable.
        """
        super().__init__()
        self.categories = categories
        self.size = len(categories)

    def denormalize(self, x: np.ndarray):
        """
        Denormalize the given normalized value for a categorical variable.

        Parameters:
        x (np.ndarray): Normalized value to be denormalized.

        Returns:
        str: Denormalized categorical value.
        """
        return self.categories[np.argmax(x)]


class Space:
    def __init__(self, vars: list[Variable]):
        """
        Initialize a Space instance with a list of variables.

        Parameters:
        vars (list[Variable]): List of variables defining the space.
        """
        self.vars = vars
        self.n = len(vars)
        self.inds = self._compute_inds(vars)
        self.size = sum([var.size for var in vars])
        self.bounds = [(0, 1) for _ in range(self.size)]

    def sample(self, n: int):
        """
        Sample n points uniformly from the space.

        Parameters:
        n (int): Number of points to sample.

        Returns:
        np.ndarray: Array of sampled points.
        """
        return np.random.uniform(0, 1, size=(n, self.size))

    def denormalize(self, xs):
        """
        Denormalize the given list of normalized values for all variables.

        Parameters:
        xs (list[np.ndarray]): List of normalized values to be denormalized.

        Returns:
        list: List of denormalized values for each variable.
        """
        return [var.denormalize(xs[ind]) for var, ind in zip(self.vars, self.inds)]

    @staticmethod
    def _compute_inds(vars: list[Variable]):
        """
        Compute the indices for each variable in the space.

        Parameters:
        vars (list[Variable]): List of variables.

        Returns:
        list[slice]: List of slices representing indices for each variable.
        """
        inds = []
        ind = 0
        for var in vars:
            inds.append(slice(ind, ind + var.size))
            ind += var.size
        return inds
