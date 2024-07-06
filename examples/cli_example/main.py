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
