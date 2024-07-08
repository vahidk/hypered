from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .aquisition_fn import AcquisitionFn
from .gaussian_process import GaussianProcess
from .kernel import Kernel
from .space import Space, Variable


def propose_location(acquisition_fn, model, space, y_opt, n_restarts=5):
    """
    Propose the next location to sample using the acquisition function.

    Parameters:
    acquisition_fn (Callable): The acquisition function to optimize.
    model (GaussianProcess): The Gaussian process model.
    space (Space): The search space.
    y_opt (float): The current optimal value of the objective function.
    n_restarts (int): The number of restarts for the optimizer.

    Returns:
    np.ndarray: The proposed location to sample, reshaped to (1, -1).
    """
    min_val = np.inf
    min_x = None

    def min_obj(x):
        return acquisition_fn(model, x.reshape(1, -1), y_opt)

    for x0 in space.sample(n_restarts):
        res = minimize(min_obj, x0=x0, bounds=space.bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(1, -1)


def bayesian_optimization(
    loss_fn: Callable,
    vars: list[Variable],
    kernel_type: str = "RBF",
    kernel_scale: float = 1.0,
    acquisition_fn_type: str = "EI",
    n_initial_points: int = 10,
    n_calls: int = 100,
    n_optimizer_restarts: int = 5,
):
    """
    Perform Bayesian optimization to minimize the given loss function.

    Parameters:
    loss_fn (Callable): The objective function to minimize.
    vars (list[Variable]): List of variables defining the search space.
    kernel_type (str): The type of kernel to use in the Gaussian process model.
    acquisition_fn_type (str): The type of acquisition function to use.
    n_initial_points (int): The number of initial points to sample.
    n_calls (int): The total number of function evaluations.
    n_optimizer_restarts (int): The number of restarts for the acquisition function optimizer.

    Returns:
    list: A list of (x, y) where x are the sampled points and ys are the corresponding function values.
    """
    kernel = Kernel.create(kernel_type, scale=kernel_scale)
    model = GaussianProcess(kernel=kernel)

    acquisition_fn = AcquisitionFn.create(acquisition_fn_type)

    space = Space(vars)

    xs_n = space.sample(n_initial_points)
    xs = [space.denormalize(x) for x in xs_n]
    ys = np.array([loss_fn(x) for x in xs])

    n_iter = n_calls - n_initial_points

    for i in range(n_iter):
        model.fit(xs_n, ys)
        y_opt = min(ys)
        x_n = propose_location(
            acquisition_fn=acquisition_fn,
            model=model,
            space=space,
            y_opt=y_opt,
            n_restarts=n_optimizer_restarts,
        )

        x = space.denormalize(x_n.flatten())
        y_n = loss_fn(x)

        xs.append(x)
        xs_n = np.append(xs_n, x_n, axis=0)
        ys = np.append(ys, y_n)

    return list(zip(xs, ys))
