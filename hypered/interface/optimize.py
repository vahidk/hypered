"""Hyperparameter optimizer interface.

This module provides a function to optimize hyperparameters using Gaussian Process minimization with `skopt`.
It supports the creation of experiments, managing directories, and executing subprocesses for the experiments.
"""

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
    binary: str,
    params: dict,
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
        binary (str): The command line binary to execute the experiment.
        params (dict): The dictionary of parameters to optimize.
        random_starts (int, optional): The number of random initialization points. Defaults to 10.
        iterations (int, optional): The number of iterations to run the optimization. Defaults to 100.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        parallelism (int, optional): The number of parallel jobs to run. Defaults to 1.
        cwd (str, optional): The current working directory for the subprocess. Defaults to None.

    Returns:
        None
    """
    print("Parameter group:", name)

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

        # Create temporary experiment directory
        m = dict_utils.hash_json(sample_params)
        experiment_dir = os.path.join(output_dir, m)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        params_path = os.path.join(experiment_dir, "params.json")
        results_path = os.path.join(experiment_dir, "results.json")

        # Call all callable functions
        wraped_params = {
            k: (
                v(
                    {
                        "name": k,
                        "params": wraped_params,
                        "experiment_dir": experiment_dir,
                        "params_path": params_path,
                        "results_path": results_path,
                    }
                )
                if callable(v)
                else v
            )
            for k, v in wraped_params.items()
        }

        # Write params to file
        with open(params_path, "w") as f:
            f.write(dict_utils.serialize_json(wraped_params, indent=4))

        # Call subprocess to perform the experiment
        if not os.path.exists(results_path):
            print("Launching experiment...")
            cmd = binary.format(params_path=params_path, results_path=results_path)
            print(cmd)
            popen = subprocess.Popen(shlex.split(cmd), cwd=cwd)
            popen.wait()
            print("Done.")
        else:
            print("Skipping experiment.")

        # Read results
        with open(results_path) as f:
            results = dict_utils.deserialize_json(f.read())

        obj_val = objective(results)

        experiments.append(
            {
                "experiment_dir": experiment_dir,
                "params_path": dict_utils.wrap_dict(sample_params),
                "results_path": results,
                "objective": obj_val,
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

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Parameter group {name}\n")
        f.write(f"Best experiment: {best['experiment_dir']}\n")
        f.write("Best results:\n")
        f.write(dict_utils.serialize_json(best["results_path"], indent=4))
        f.write("\nBest params:\n")
        f.write(dict_utils.serialize_json(best["params_path"], indent=4))
        f.write("\n")

    print(open(os.path.join(output_dir, "results.txt")).read())
