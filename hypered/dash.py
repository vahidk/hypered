import argparse
import logging

import flask

from .interface.misc import OUTPUT_DIR
from .server.experiment_loader import ExperimentLoader

app = flask.Flask(__name__)
el = ExperimentLoader(OUTPUT_DIR)


@app.route("/")
def index():
    el.load_experiments()
    return flask.render_template("index.html")


@app.route("/experiment_groups", methods=["GET"])
def get_experiment_groups():
    return flask.jsonify(list(el.experiment_data.keys()))


@app.route("/experiment_group/<group_name>/names", methods=["GET"])
def get_experiment_group_names(group_name: str):
    experiments = el.experiment_data[group_name]
    variables = list(experiments["params"].keys())
    metrics = list(experiments["results"].keys())
    return flask.jsonify({"variables": variables, "metrics": metrics})


@app.route("/experiment_group/<group_name>/data", methods=["GET"])
def get_experiment_group_data(group_name: str):
    experiments = el.experiment_data[group_name]
    variables = experiments["params"]
    metrics = experiments["results"]
    if experiments["best"]:
        best = experiments["best"]
        best = {"variables": best["params"], "metrics": best["results"]}
    else:
        best = None
    return flask.jsonify({"variables": variables, "metrics": metrics, "best": best})


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer dashboard.")
    parser.add_argument("--host", type=str, default=None, help="HTTP server host.")
    parser.add_argument("--port", type=int, default=None, help="HTTP server port.")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
