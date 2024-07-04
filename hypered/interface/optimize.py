"""Hyperparameter optimizer interface.

This module provides a function to optimize hyperparameters using Gaussian Process minimization with `skopt`.
It supports the creation of experiments, managing directories, and executing subprocesses for the experiments.
"""

import logging
import os
import shlex
import subprocess
from typing import Callable, Optional

import skopt

from . import misc, variable
from .common import dict_utils, registry


@registry.export
def optimize(
    name: str,
    objective: Callable,
    params: dict,
    binary: Optional[str] = None,
    function: Optional[Callable] = None,
    random_starts: int = 10,
    iterations: int = 100,
    seed: int = 0,
    parallelism: int = 1,
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
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        parallelism (int, optional): The number of parallel jobs to run. Defaults to 1.
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

    unwraped_params = dict_utils.unwrap_dict(params)

    # Extract all variables
    keys = []
    space = []
    for k, v in unwraped_params.items():
        if isinstance(v, variable.variable):
            keys.append(k)
            space.append(v())

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
        merged_params = dict_utils.merge_dicts(unwraped_params, sample_params)
        wraped_params = dict_utils.wrap_dict(merged_params)

        logging.info(
            "Evaluating parameters: %s",
            dict_utils.serialize_json(sample_params, indent=4),
        )

        # Create temporary experiment directory
        if binary is not None:
            experiment_dir = os.path.join(
                output_dir, dict_utils.hash_json(sample_params)
            )
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
                f.write(dict_utils.serialize_json(wraped_params, indent=4))

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
                results = dict_utils.deserialize_json(f.read())
        else:
            raise ValueError("Either binary or function must be provided.")

        obj_val = objective(results)

        experiments.append(
            {
                "params": dict_utils.wrap_dict(sample_params),
                "results": results,
                "objective": obj_val,
                **extra_params,
            }
        )

        return obj_val

    res: skopt.OptimizeResult = skopt.gp_minimize(
        _eval,
        space,
        n_random_starts=random_starts,
        n_calls=iterations,
        random_state=seed,
        n_jobs=parallelism,
    )

    # Find the best experiment results
    best = experiments[0]
    for exp in experiments:
        if exp["objective"] < best["objective"]:
            best = exp
    assert res.fun == best["objective"]

    if binary is not None:
        results_path = os.path.join(output_dir, "results.txt")
        summary = "\n".join([
            f"Parameter group {name}",
            f"Best experiment: {best['experiment_dir']}",
            "Best results:",
            dict_utils.serialize_json(best["results"], indent=4),
            "Best params:",
            dict_utils.serialize_json(best["params"], indent=4)
        ])
        logging.info(f"Writing results to {results_path}:\n{summary}")
        with open(results_path, "w") as f:
            f.write(summary)
    else:
        return best
