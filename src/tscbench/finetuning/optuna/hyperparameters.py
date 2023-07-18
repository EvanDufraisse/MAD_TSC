#
# Created on Wed Jul 06 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
from codecs import ignore_errors
from xml.dom.pulldom import IGNORABLE_WHITESPACE
import optuna
import os
from tscbench.finetuning.optuna.customgridsampler import CustomGridSampler
from abc import ABC, abstractmethod
from optuna.trial import TrialState


def get_size_search_space(search_space: dict):
    """
    Get the size of the search space
    """
    size = 1
    for key, value in search_space.items():
        size = size * len(value)
    return size


class Pipeline(ABC):
    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.objective = None
        self.search_space = None

    def get_objective(self):
        return self.objective

    def get_search_space(self) -> "dict[str, list]":
        """
        Get the search space as a dictionary of lists of values
        """
        return self.search_space

    def set_seeds(self, seeds):
        self.search_space["seeds"] = seeds


class HyperparameterSelection(object):
    """
    This class allows to select the top k hyperparameters ensembles for a given pipeline
    """

    def __init__(
        self,
        pipeline: Pipeline,
        path_optuna_db: str,
        study_name: str,
        k: int = 3,
        directions=["max"],
        force_distinct_hyper: bool = True,
        max_fail_trials_per_grid: int = 1,  # max number of retry before skipping a failed grid (1 = one retry)
        ignore_hyperparameter: "list[str]" = [],
    ):
        self.pipeline = pipeline
        self.k = k
        self.directions = [
            (
                optuna.study.StudyDirection.MAXIMIZE
                if "max" in elem
                else optuna.study.StudyDirection.MINIMIZE
            )
            for elem in directions
        ]
        # print(self.directions)
        self.force_distinct_hyper = force_distinct_hyper
        self.path_optuna_db = path_optuna_db
        self.study_name = study_name
        self.max_fail_trials_per_grid = max_fail_trials_per_grid
        self.ignore_hyperparameter = set(ignore_hyperparameter)

        os.makedirs(
            os.path.dirname(self.path_optuna_db), exist_ok=True
        )  # create the directory if it does not exist

        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.path_optuna_db}",
        )  # create the storage

        self.search_space = (
            self.pipeline.get_search_space()
        )  # get the search space dictionary

        # assert self.k <= get_size_search_space(
        # self.search_space
        # )  # check that the number of top k runs to keep is not greater than the size of the search space

        if len(self.ignore_hyperparameter) > 0 and self.force_distinct_hyper:
            temp_search_space = {
                key: value
                for key, value in self.search_space.items()
                if key not in self.ignore_hyperparameter
            }  # remove the hyperparameters to ignore from the search space
            size_not_ignored_search_space = get_size_search_space(temp_search_space)
            assert self.k < size_not_ignored_search_space

        self.study = optuna.create_study(
            study_name=self.study_name,
            directions=self.directions,
            sampler=CustomGridSampler(
                search_space=self.search_space,
                max_fail_trials_per_grid=self.max_fail_trials_per_grid,
                path_lock_db=os.path.join(self.path_optuna_db.split(".")[0] + ".lock"),
            ),
            storage=storage,
            load_if_exists=True,
        )

        self.size_search_space = get_size_search_space(self.search_space)
        self.n_trials = self.size_search_space * (max_fail_trials_per_grid + 1)

    def optimize(self):
        self.study.optimize(self.pipeline.get_objective(), n_trials=self.n_trials)

    def get_top_k_results(self, k=None, index=0) -> "list[tuple[dict, TrialState]]":
        """Returns the top k results of the study

        Returns:
            list[tuple[dict, TrialState]]: _description_
        """

        if not (k is None):
            k = k
        else:
            k = self.k
        sign = (
            -1
            if self.study.directions[index] == optuna.study.StudyDirection.MAXIMIZE
            else 1
        )
        completed_trials = sorted(
            [
                (trial, trial.values[index])
                for trial in self.study.trials
                if trial.state == TrialState.COMPLETE
                and not (trial.values is None)
                and len(trial.values) > 0
            ],
            key=lambda x: sign * x[-1],
        )

        trials, _ = zip(*completed_trials)
        rank_mapping = {
            i: (
                {
                    key: value
                    for key, value in trial.params.items()
                    if not (key in self.ignore_hyperparameter)
                },
                trial,
            )
            for i, trial in enumerate(trials)
        }
        if not (self.force_distinct_hyper):
            return [
                rank_mapping[i] for i in range(k)
            ]  # return the top k results with hyperparameters without seed and whole trial
        else:
            output = []
            already_added_config = set([])
            idx = 0
            while len(output) < k:
                if not (k in rank_mapping):
                    break
                config, trial = rank_mapping[idx]
                str_config = str(sorted(list(config.items()), key=lambda x: x[0]))
                if str_config not in already_added_config:
                    output.append((config, trial))
                    already_added_config.add(str_config)
                idx += 1
            return output

    def get_all_results(self):
        """Returns all the results of the study

        Returns:
            list[tuple[dict, TrialState]]: _description_
        """
        return [
            (trial.params, trial)
            for trial in self.study.trials
            if trial.state == TrialState.COMPLETE
            and not (trial.values is None)
            and len(trial.values) > 0
        ]
