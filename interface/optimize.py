"""Hyper parameter optimizer interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import skopt
import subprocess
import shlex
import os

from common import utils
from interface import misc
from interface import registry
from interface import variable


@registry.export
def optimize(
  name, objective, binary, params, random_starts=10,
  iterations=100, seed=0, parallelism=1, cwd=None):

  print("Parameter group:", name)

  output_dir = os.path.abspath(os.path.join(misc.OUTPUT_DIR, name))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  unwraped_params = utils.unwrap_dict(params)

  # Extract all variables
  keys = []
  space = []
  for k, v in unwraped_params.items():
    if isinstance(v, variable.variable):
      keys.append(k)
      space.append(v())

  experiments = []

  def _eval(values):
    # Merge base parameters with sampled params
    sample_params = dict(zip(keys, values))
    merged_params = utils.merge_dicts(unwraped_params, sample_params)
    wraped_params = utils.wrap_dict(merged_params)

    # Create temprary experiment directory
    m = utils.hash_json(sample_params)
    experiment_dir = os.path.join(output_dir, m)
    if not os.path.exists(experiment_dir):
      os.makedirs(experiment_dir)

    params_path = os.path.join(experiment_dir, "params.json")
    results_path = os.path.join(experiment_dir, "results.json")

    # Call all callable functions
    wraped_params = {
      k: v({
        "name": k,
        "params": wraped_params,
        "experiment_dir": experiment_dir,
        "params_path": params_path,
        "results_path": results_path
      }) if callable(v) else v 
      for k, v in wraped_params.items()
    }

    # Write params to file
    with open(params_path, "w") as f:
      f.write(utils.serialize_json(wraped_params, indent=4))

    # Call subprocess to perform the experiment
    if not os.path.exists(results_path):
      print("Launching experiment...")
      cmd = binary.format(
        params_path=params_path, 
        results_path=results_path)
      print(cmd)
      popen = subprocess.Popen(shlex.split(cmd), cwd=cwd)
      popen.wait()
      print("Done.")
    else:
      print("Skipping experiment.")

    # Read results
    with open(results_path) as f:
      results = utils.deserialize_json(f.read())

    obj_val = objective(results)

    experiments.append({
      "experiment_dir": experiment_dir,
      "params_path": utils.wrap_dict(sample_params),
      "results_path": results,
      "objective": obj_val
    })
    
    return obj_val

  res = skopt.gp_minimize(
    _eval, space, n_random_starts=random_starts, n_calls=iterations, 
    random_state=seed, n_jobs=parallelism)
  
  # Find the best experiment results
  best = experiments[0]
  for exp in experiments:
    if exp["objective"] < best["objective"]:
      best = exp
  assert(res.fun == best["objective"])

  with open(os.path.join(output_dir, "results.txt"), "w") as f:
    f.write(f"Parameter group {name}\n")
    f.write(f"Best experiment: {best["experiment_dir"]}\n")
    f.write("Best results:\n")
    f.write(utils.serialize_json(best["results_path"], indent=4))
    f.write("\nBest params:\n")
    f.write(utils.serialize_json(best["params_path"], indent=4))
    f.write("\n")
  
  print(open(os.path.join(output_dir, "results.txt")).read())
