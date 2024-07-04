import json
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from hypered.interface.common import dict_utils


class Watcher(FileSystemEventHandler):
    experiment_data = {}

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.load_experiments()
        self.observer = self.create_observer()

    def create_observer(self):
        observer = Observer()
        observer.schedule(self, path=self.directory, recursive=True)
        observer.start()

    def on_modified(self, event):
        if event.is_directory:
            return
        self.load_experiments()

    def load_experiments(self):
        self.experiment_data = {}
        for experiment_group in os.listdir(self.directory):
            group_data = self.load_experiment_group(experiment_group)
            if group_data is None:
                continue
            self.experiment_data[experiment_group] = group_data

    def load_experiment_group(self, experiment_group):
        group_path = os.path.join(self.directory, experiment_group)
        if not os.path.isdir(group_path):
            return None
        params_list = []
        results_list = []
        for experiment in os.listdir(group_path):
            experiment_path = os.path.join(group_path, experiment)
            params, results = self.load_experiment(experiment_path)
            if params is None or results is None:
                continue
            params_list.append(params)
            results_list.append(results)
        params = dict_utils.join_dicts(params_list)
        results = dict_utils.join_dicts(results_list)
        return {"params": params, "results": results}

    def load_experiment(self, experiment_path: str):
        params_path = os.path.join(experiment_path, "params.json")
        results_path = os.path.join(experiment_path, "results.json")
        if not os.path.exists(params_path) or not os.path.exists(results_path):
            return None, None
        with open(params_path) as params_file:
            params = json.load(params_file)
        with open(results_path) as results_file:
            results = json.load(results_file)
        return params, results
