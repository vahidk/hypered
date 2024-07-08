import json
import os

from ..utils.dict_utils import join_dicts


class ExperimentLoader:
    experiment_data = {}

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def load_experiments(self):
        self.experiment_data = {}
        if not os.path.exists(self.directory):
            print("Directory does not exist")
            return
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
        params = join_dicts(params_list)
        results = join_dicts(results_list)
        best_path = os.path.join(group_path, "best.json")
        if os.path.exists(best_path):
            with open(best_path) as best_file:
                best = json.load(best_file)
        else:
            best = None
        return {"params": params, "results": results, "best": best}

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
