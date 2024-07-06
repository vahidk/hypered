"""Example model."""

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


def main(params_path: str, results_path: str):
    params = json.loads(open(params_path).read())
    results = eval_objective(params)
    with open(results_path, "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple model.")
    parser.add_argument("params", type=str, help="Params file.")
    parser.add_argument("results", type=str, help="Results file.")
    args = parser.parse_args()
    main(args.params, args.results)
