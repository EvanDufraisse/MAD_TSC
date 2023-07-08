# -*- coding: utf-8 -*-
""" AbsaPipeline Class

The AbsaPipeline class implements the Pipeline object to return scores to the HyperparameterSearch based on Optuna. 

@Author: Evan Dufraisse
@Date: Fri Nov 25 2022
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2022 CEA - LASTI
"""

from tscbench.finetuning.optuna.hyperparameters import Pipeline
from tscbench.finetuning.absa.objective import AbsaOptunaObjective


class AbsaPipeline(Pipeline):
    def __init__(
        self,
        pipeline_name,
        model_path,
        tokenizer_path,
        absa_config,
        dataset_config,
        gpu_pl_config,
        optimizer_config,
        sub_path_final_folder,
        storage_manager,
        path_final_output_model,
        path_scratch_output_model,
        uuid,
        ckpt_path=None,
        other_args={},
        keep_best_models=False,
        keep_all_models=False,
        custom_datasets_json=None,
    ):
        self.pipeline_name = pipeline_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.absa_config = absa_config
        self.dataset_config = dataset_config
        self.gpu_pl_config = gpu_pl_config
        self.optimizer_config = optimizer_config
        self.sub_path_final_folder = sub_path_final_folder
        self.uuid = uuid
        self.storage_manager = storage_manager
        self.path_final_output_model = path_final_output_model
        self.path_scratch_output_model = path_scratch_output_model
        self.seeds = self.optimizer_config["optimizer"]["hyperparameters"]["seeds"]
        self.ckpt_path = ckpt_path
        self.other_args = other_args
        self.keep_best_models = keep_best_models
        self.keep_all_models = keep_all_models
        self.custom_datasets_json = custom_datasets_json

    def get_objective(self):
        """Returns the objective function to be optimized.

        Returns:
            AbsaOptunaObjective: _description_
        """
        return AbsaOptunaObjective(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            absa_config=self.absa_config,
            dataset_config=self.dataset_config,
            gpu_pl_config=self.gpu_pl_config,
            optimizer_config=self.optimizer_config,
            subpath_final_folder=self.sub_path_final_folder,
            uuid=self.uuid,
            storage_manager=self.storage_manager,
            path_final_output_model=self.path_final_output_model,
            path_scratch_output_model=self.path_scratch_output_model,
            ckpt_path=self.ckpt_path,
            other_args=self.other_args,
            keep_best_models=self.keep_best_models,
            keep_all_models=self.keep_all_models,
            custom_datasets_json=self.custom_datasets_json,
        )

    def get_search_space(self):
        """Returns the hyperparameter search space.

        Returns:
            dict: dictionary of hyperparameters
        """
        hyperparameters = self.optimizer_config["optimizer"]["hyperparameters"]
        return hyperparameters

    def set_seeds(self, seeds):
        """Sets the 'seeds' field of the self.search_space dictionnary of the pipeline."""
        return super().set_seeds(seeds)
