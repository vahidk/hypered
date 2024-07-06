"""
This is an example of a hyperparameter optimization script using the `hypered` library.

To run this example, use the following command:
```bash
python main.py
```

"""

import json

import numpy as np

import hypered as hp


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
    params = hp.optimize(
        name="learning_params",
        objective=hp.minimize("loss"),
        function=eval_objective,
        random_starts=10,
        iterations=30,
        parallelism=8,
        params={
            "option": hp.categorical(["first", "second"]),
            "x": hp.real(-10, 10),
        },
    )
    print(json.dumps(params, indent=4))


if __name__ == "__main__":
    main()
