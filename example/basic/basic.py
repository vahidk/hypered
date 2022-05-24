"""Example model."""

import argparse
import json
import numpy as np


def main(args):
  params = json.loads(open(args.params).read())

  op = params["vars"]["option"]
  x = params["vars"]["x"]

  if op == "first":
    loss = np.square(x - 5)
  elif op == "second":
    loss = np.abs(x - 3) - 2
  else:
    print("Invalid option", op)
    exit(0)

  print(x, loss)
  results = {
    "loss": loss
  }
  with open(args.results, "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Simple model.")
  parser.add_argument("params", type=str, help="Params file.")
  parser.add_argument("results", type=str, help="Results file.")
  args = parser.parse_args()
  main(args)
