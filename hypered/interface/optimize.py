"""Hyperparameter optimizer interface.

This module provides a function to optimize hyperparameters using Gaussian Processes.
It supports the creation of experiments, managing directories, and executing subprocesses for the experiments.
"""

import logging
import os
import shlex
import subprocess
from typing import Callable, Optional

from . import misc, variable
from .registry import export
from ..optim.bayesian_optimization import bayesian_optimization
from ..utils.dict_utils import (
    merge_dicts,
    unwrap_dict,
    wrap_dict,
    serialize_json,
    deserialize_json,
    hash_json,
)


@export
def optimize(
    name: str,
    objective: Callable,
    params: dict,
    binary: Optional[str] = None,
    function: Optional[Callable] = None,
    random_starts: int = 10,
    iterations: int = 100,
    kernel: str = "RBF",
    kernel_scale: float = 1.0,
    acquisition_fn: str = "EI",
    optimizer_restarts: int = 5,
    cwd: Optional[str] = None,
):
    """
    Optimize hyperparameters using Gaussian Process minimization.

    Args:
        name (str): The name of the parameter group.
        objective (function): The objective function to minimize. It should take a dictionary of results and return a scalar value.
        params (dict): The dictionary of parameters to optimize.
        binary (str, optional): The command line binary to execute the experiment.
        function (function, optional): The function to execute the experiment.
        random_starts (int, optional): The number of random initialization points. Defaults to 10.
        iterations (int, optional): The number of iterations to run the optimization. Defaults to 100.
        kernel (str, optional): The type of kernel to use in the Gaussian process model. Defaults to "RBF".
        kernel_scale (float, optional): The scale of the kernel. Defaults to 1.0.
        acquisition_fn (str, optional): The type of acquisition function to use. Defaults to "EI".
        optimizer_restarts (int, optional): The number of restarts for the optimizer. Defaults to 5.
        cwd (str, optional): The current working directory for the subprocess. Defaults to None.

    Returns:
        None
    """
    if binary is None and function is None:
        raise ValueError("Either binary or function must be provided.")

    logging.info("Parameter group: %s", name)

    if binary is not None:
        output_dir = os.path.abspath(os.path.join(misc.OUTPUT_DIR, name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    unwraped_params = unwrap_dict(params)

    # Extract all variables
    keys = []
    vars = []
    for k, v in unwraped_params.items():
        if isinstance(v, variable.variable):
            keys.append(k)
            vars.append(v())

    experiments = []

    def _eval(values: list):
        """
        Evaluate the objective function with the given parameter values.

        Args:
            values (list): The list of parameter values to evaluate.

        Returns:
            float: The value of the objective function for the given parameter values.
        """
        # Merge base parameters with sampled params
        sample_params = dict(zip(keys, values))
        merged_params = merge_dicts(unwraped_params, sample_params)
        wraped_params = wrap_dict(merged_params)

        logging.info(
            "Evaluating parameters: %s",
            serialize_json(sample_params, indent=4),
        )

        # Create temporary experiment directory
        if binary is not None:
            experiment_dir = os.path.join(output_dir, hash_json(sample_params))
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            params_path = os.path.join(experiment_dir, "params.json")
            results_path = os.path.join(experiment_dir, "results.json")
        else:
            experiment_dir = ""
            params_path = ""
            results_path = ""

        extra_params = {
            "experiment_dir": experiment_dir,
            "params_path": params_path,
            "results_path": results_path,
        }

        # Call all callable functions
        wraped_params = {
            k: (
                v({"name": k, "params": wraped_params, **extra_params})
                if callable(v)
                else v
            )
            for k, v in wraped_params.items()
        }

        if function is not None:
            results = function(wraped_params)
        elif binary is not None:
            # Write params to file
            with open(params_path, "w") as f:
                f.write(serialize_json(wraped_params, indent=4))

            # Call subprocess to perform the experiment
            if not os.path.exists(results_path):
                logging.info("Launching experiment...")
                cmd = binary.format(params_path=params_path, results_path=results_path)
                logging.info(cmd)
                popen = subprocess.Popen(shlex.split(cmd), cwd=cwd)
                popen.wait()
                logging.info("Done.")
            else:
                logging.info("Skipping experiment.")

            # Read results
            with open(results_path) as f:
                results = deserialize_json(f.read())
        else:
            raise ValueError("Either binary or function must be provided.")

        loss_val = objective(results)

        experiments.append(
            {
                "params": wrap_dict(sample_params),
                "results": results,
                "loss": loss_val,
                **extra_params,
            }
        )

        return loss_val

    bayesian_optimization(
        _eval,
        vars,
        kernel_type=kernel,
        kernel_scale=kernel_scale,
        acquisition_fn_type=acquisition_fn,
        n_initial_points=random_starts,
        n_calls=iterations,
        n_optimizer_restarts=optimizer_restarts,
    )

    # Find the best experiment results
    best = experiments[0]
    for exp in experiments:
        if exp["loss"] < best["loss"]:
            best = exp

    if binary is not None:
        summary_path = os.path.join(output_dir, "best.json")
        summary = serialize_json(
            {
                "experiment_dir": best["experiment_dir"],
                "params": best["params"],
                "results": best["results"],
            }
        )
        logging.info(f"Writing results to {summary_path}:\n{summary}")
        with open(summary_path, "w") as f:
            f.write(summary)
    else:
        return best
