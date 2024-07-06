import argparse
import json
import logging
import os

import flask

from .interface import misc
from .server import watcher

app = flask.Flask(__name__)
watch = watcher.Watcher(misc.OUTPUT_DIR)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/experiment_groups", methods=["GET"])
def get_experiment_groups():
    return flask.jsonify(list(watch.experiment_data.keys()))


@app.route("/experiment_group/<group_name>/names", methods=["GET"])
def get_experiment_group_names(group_name):
    experiments = watch.experiment_data[group_name]
    variables = list(experiments["params"]["vars"].keys())
    metrics = list(experiments["results"].keys())
    return flask.jsonify({"variables": variables, "metrics": metrics})


@app.route("/experiment_group/<group_name>/data", methods=["GET"])
def get_experiment_group_data(group_name):
    experiments = watch.experiment_data[group_name]
    variables = experiments["params"]["vars"]
    metrics = experiments["results"]
    return flask.jsonify({"variables": variables, "metrics": metrics})


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer.")
    parser.add_argument("--host", type=str, default=None, help="HTTP server host.")
    parser.add_argument("--port", type=int, default=None, help="HTTP server port.")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
