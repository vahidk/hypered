<h1 align="center">
<img src="https://raw.githubusercontent.com/vahidk/hypered/main/media/hypered.png" alt="SafeConfig Library" width="256">
</h1><br>


[![PyPI - Downloads](https://img.shields.io/pypi/dm/hypered)](https://pypi.org/project/hypered/)

Hypered provides a flexible interface for optimizing hyperparameters of any blackbox system. It implements bayesian optimization and supports the creation of various types of hyperparameter search spaces including real, integer, and categorical variables.

## Features

- **Hyperparameter Spaces**: Define real, integer, and categorical variables.
- **Objective Functions**: Easily create minimization and maximization objectives.
- **Experiment Management**: Automatically handles experiment directories and parameter/result files.
- **Web-based Dashboard**: Visualize the experiment results for better insight.

## Installation

To install hypered, simply run:

```bash
pip install hypered
```

## Usage

### Step 1: Model Script

To use hypered you first need to define a model script that takes the hyper-parameters as an input json file and computes the loss/objective value and write it as a json file. Both input and output files should be provided by a command line arguments. The following is an example model script:

**example.py**
```python
import argparse
import json
import numpy as np

def eval_objective(params: dict) -> dict:
    op = params["option"]
    x = params["x"]

    if op == "first":
        loss = np.square(x - 5)
    elif op == "second":
        loss = np.abs(x - 3) - 2
    else:
        print("Invalid option", op)
        exit(0)

    return {"loss": loss}

def main():
    parser = argparse.ArgumentParser(description="Simple model.")
    parser.add_argument("params", type=str, help="Params file.")
    parser.add_argument("results", type=str, help="Results file.")
    args = parser.parse_args()

    params = json.loads(open(args.params).read())
    results = eval_objective(params)
    with open(args.results, "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    main()
```

### Step 2: Configuration File

Next we need to define a configuration file that specifies the hyper parameters as well as the objective function. Below is an example configuration file:

**example.conf**
```python
optimize(
    name="learning_params",
    objective=minimize("loss"),
    binary="python3 example.py {params_path} {results_path}",
    random_starts=10,
    iterations=30,
    params={
        "option": categorical(["first", "second"]),
        "x": real(-10, 10)
    }
)
```

### Step 3: Running the Hyperparameter Optimizer

To run the hyperparameter optimizer, use the `hypered` script with the path to your configuration file:

```bash
hypered example.conf
```

This will start the optimization process as defined in your configuration file and output the best parameters.

### Step 4: Visualize the Results on Dashboard

Finally, you can visualize the results of hyper-parameter optimization on hypered-dash with the following command:

```bash
hypered-dash
```

## Reference

### Optimizers

#### `optimize`

This function performs hyperparameter optimization using Gaussian Processes.

**Arguments:**
- `name` (str): The name of the parameter group.
- `objective` (function): The objective function to minimize or maximize.
- `binary` (str): The command line binary to execute the experiment.
- `params` (dict): The dictionary of parameters to optimize.
- `random_starts` (int, optional): The number of random initialization points.
- `iterations` (int, optional): The number of iterations to run the optimization.
- `kernel` (str, optional): The type of kernel to use in the Gaussian process model. Defaults to "RBF".
- `kernel_scale` (float, optional): The scale of the kernel. Defaults to 1.0.
- `acquisition_fn` (str, optional): The type of acquisition function to use. Defaults to "EI".
- `optimizer_restarts` (int, optional): The number of restarts for the optimizer. Defaults to 5.
- `seed` (int, optional): The random seed for reproducibility.
- `cwd` (str, optional): The current working directory for the subprocess.

Note that you can use predefined variables {params_path} and {results_path} in your binary string to specify the path to parameters and results json files accordingly.

### Objectives

#### `minimize`

Minimize a given variable.

#### `maximize`

Maximize a given variable.

### Variables

#### `real`

Class for defining a real-valued hyperparameter.

#### `integer`

Class for defining an integer-valued hyperparameter.

#### `categorical`

Class for defining a categorical hyperparameter.

### Utilities

#### `experiment_dir`

Retrieves the experiment directory from the context.

#### `params_path`

Retrieves the parameters path from the context.

#### `results_path`

Retrieves the results path from the context.

#### `device_id`

Returns a device ID in a round-robin fashion.

## License

This library is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
