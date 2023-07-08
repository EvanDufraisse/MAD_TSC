# -*- coding: utf-8 -*-
""" Fine Tuner module

@Author: Evan Dufraisse
@Date: Thu May 04 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
import torch
import os
import pytorch_lightning as pl
import json
import os
from tscbench.utils.storage import ExperimentStorageManager
import logging
from tscbench.finetuning.optuna.customgridsampler import (
    NoGridLeftToVisit,
)
from tscbench.finetuning.optuna.hyperparameters import HyperparameterSelection
from tscbench.finetuning.absa.pipeline import AbsaPipeline
import sqlite3
import time
from loguru import logger


class FineTuner(object):
    """This class organizes the finetuning and the parameter seach of a model"""

    def __init__(
        self,
        name_experiment: str,
        model_path: str,
        tokenizer_path: str,
        absa_config_path,
        dataset_config_path,
        gpu_pl_config_path,
        optimizer_config_path,
        subpath_final_folder,
        max_batch_size_per_gpu=16,
        uuid=None,
        ckpt_path=None,
        other_args={},
        keep_best_models=False,
        keep_all_models=False,
        custom_datasets_json=None,
        experiment_dir=None,
        data_dir=None,
        experiment_scratch=None,
        data_scratch=None,
    ):
        """

        Loads the configuration jsons supplied.
        Modifies the maximum batch size depending on the slurm node we get.
        Modifies the maximum batch size if the model trained is a multi-prompt model

        Args:
            name_experiment (str): name of experiment, as should appear as identifying folder
            model_path (str): path to the huggingface model to use as mlm model
            tokenizer_path (str): path to the huggingface tokenizer
            absa_config_path (str): path to the configuration json of the model
            dataset_config_path (str): path to the configuration json of the dataset
            gpu_pl_config_path (str): path to the configuration of pytorch lightning (pl) for training
            optimizer_config_path (str): path to the configuration of optimization, parameters and layers to use
            subpath_final_folder (str): final folder is determined using the $EXPERIMENT_DIR env variable
            uuid (str, optional): unique identifier that serves to create temporary folders unique to the experiment. Defaults to None.
            ckpt_path (str, optional): To load existing weight of model. Defaults to None.
            other_args (dict, optional): Other args you can pass to the model constructor. Defaults to {}.
            keep_best_models (bool, optional): Wether to keep or not the weights of the best performing model in the final folder. Defaults to False.
            old_training_version (bool, optional): Run older version of finetuning optimisation. Defaults to False.
        """

        # Load configuration jsons
        self.name_experiment = name_experiment
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.subpath_final_folder = subpath_final_folder
        self.uuid = uuid
        self.ckpt_path = ckpt_path
        self.other_args = other_args
        self.keep_best_models = keep_best_models
        self.keep_all_models = keep_all_models
        self.custom_datasets_json = custom_datasets_json
        self.max_batch_size_per_gpu = max_batch_size_per_gpu
        self.experiment_dir = experiment_dir
        self.data_dir = data_dir
        self.experiment_scratch = experiment_scratch
        self.data_scratch = data_scratch
        if self.keep_all_models:
            logger.info("All models will be kept")
        elif self.keep_best_models:
            logger.info("Only best models will be kept")
        else:
            logger.info("No models will be kept")

        self.__load_json_configuration_files(
            absa_config_path,
            dataset_config_path,
            gpu_pl_config_path,
            optimizer_config_path,
        )

        logger.info("Experiment Dir: " + str(self.experiment_dir))
        logger.info("Data Dir: " + str(self.data_dir))
        logger.info("Experiment Scratch: " + str(self.experiment_scratch))
        logger.info("Data Scratch: " + str(self.data_scratch))

        # This allows to modify the default maximum batch size depending on the node obtained through slurm

        self.__change_batch_size_function_of_gpu_obtained()

        logger.info(
            f"Max batch size per gpu: {self.optimizer_config['optimizer']['max_batch_size_per_gpu']}"
        )

        # If the model has several prompts (multi-prompt model), then we need to divide the maximum batch size by the number of prompts and do gradient accumulation
        self.__change_batch_size_if_multi_prompts_model()

    def __change_batch_size_if_multi_prompts_model(self):
        """If the model has several prompts (multi-prompt model), then we need to divide the maximum batch size by the number of prompts and do gradient accumulation"""
        if self.absa_config["absa_model"] == "pm":
            if "prompt_template" in self.absa_config["other_args"]:
                if len(self.absa_config["other_args"]["prompt_template"]) > 1:
                    self.optimizer_config["optimizer"]["max_batch_size_per_gpu"] = max(
                        1,
                        self.optimizer_config["optimizer"]["max_batch_size_per_gpu"]
                        // len(self.absa_config["other_args"]["prompt_template"]),
                    )

    def __change_batch_size_function_of_gpu_obtained(self):
        """Change the maximum batch size depending on the slurm node we get"""
        # print("Optimizer config: ", self.optimizer_config)
        if "SLURM_JOB_PARTITION" in os.environ and not (
            "max_batch_size_per_gpu" in self.optimizer_config["optimizer"]
        ):
            if os.environ["SLURM_JOB_PARTITION"] in ["lasti", "gpuv100"] and os.environ[
                "SLURMD_NODENAME"
            ] not in ["node28"]:
                self.optimizer_config["optimizer"]["max_batch_size_per_gpu"] = 32
            elif os.environ["SLURM_JOB_PARTITION"] in ["gpu"]:
                self.optimizer_config["optimizer"]["max_batch_size_per_gpu"] = 8
            else:
                self.optimizer_config["optimizer"]["max_batch_size_per_gpu"] = 16
        else:
            self.optimizer_config["optimizer"][
                "max_batch_size_per_gpu"
            ] = self.max_batch_size_per_gpu

    def __load_json_configuration_files(
        self,
        absa_config_path,
        dataset_config_path,
        gpu_pl_config_path,
        optimizer_config_path,
    ):
        """Load the json configuration files

        Args:
            absa_config_path (str): configuration of the model
            dataset_config_path (str): configuration of the dataset
            gpu_pl_config_path (str): configuration of pytorch lightning (pl) for training
            optimizer_config_path (str): configuration of optimization, parameters and layers to use
        """
        self.absa_config = json.load(open(absa_config_path, "r"))
        self.dataset_config = json.load(open(dataset_config_path, "r"))
        self.gpu_pl_config = json.load(open(gpu_pl_config_path, "r"))
        self.optimizer_config = json.load(open(optimizer_config_path, "r"))

    def create_folders_experiment(self):
        """
        Generates the temporary and definitive folders depending on the environment variables:
        $DATA_DIR
        $DATA_SCRATCH
        $EXPERIMENT_DIR
        $EXPERIMENT_SCRATCH
        and the uuid
        """
        self.storage_manager = ExperimentStorageManager(
            self.subpath_final_folder,
            self.uuid,
            experiment_dir=self.experiment_dir,
            data_dir=self.data_dir,
            experiment_scratch=self.experiment_scratch,
            data_scratch=self.data_scratch,
        )
        self.uuid = self.storage_manager.uuid
        self.storage_manager.generate_folders()
        self.path_final_output_model = self.storage_manager.get_path(
            self.name_experiment, "final"
        )
        logging.critical(self.path_final_output_model)
        self.path_scratch_output_model = self.storage_manager.get_path(
            self.name_experiment, "escratch"
        )
        self.dataset_config["dataset_temp_root_folder"] = self.storage_manager.get_path(
            "", "dscratch"
        )

    def create_pipeline_best_topk(self):
        self.pipeline = AbsaPipeline(
            pipeline_name=self.name_experiment,
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            absa_config=self.absa_config,
            dataset_config=self.dataset_config,
            gpu_pl_config=self.gpu_pl_config,
            optimizer_config=self.optimizer_config,
            sub_path_final_folder=self.subpath_final_folder,
            uuid=self.uuid,
            storage_manager=self.storage_manager,
            path_final_output_model=self.path_final_output_model,
            path_scratch_output_model=self.path_scratch_output_model,
            other_args=self.other_args,
            ckpt_path=self.ckpt_path,
            keep_best_models=self.keep_best_models,
            keep_all_models=self.keep_all_models,
            custom_datasets_json=self.custom_datasets_json,
        )

    def get_best_topk_hyperparams(self, k=5):
        path_optuna_db = self.storage_manager.get_path(
            f"{self.name_experiment}_optuna.db", "final"
        )
        # path_lock_db = self.storage_manager.get_path(
        #     f"{self.name_experiment}_optuna_lock.db", "final"
        # )
        optuna_metrics = self.optimizer_config["optimizer"]["optuna_metrics"]
        self.hp_select = HyperparameterSelection(
            pipeline=self.pipeline,
            path_optuna_db=path_optuna_db,
            study_name=f"best_top{k}_{self.name_experiment}",
            directions=[direction for direction in optuna_metrics.values()],
            k=k,
            force_distinct_hyper=True,
        )

        try:
            self.hp_select.optimize()
        except NoGridLeftToVisit as e:
            print(e)
            pass
        top_k_results = self.hp_select.get_top_k_results()
        return top_k_results

    def run(self):
        self.name_experiment = self.name_experiment + "_exp"
        logger.info("Creating folders experiment")
        self.create_folders_experiment()
        logger.info("Creating pipeline experiment")
        self.create_pipeline_best_topk()
        logger.info("Running experiment")
        top_k_results = self.get_best_topk_hyperparams()

        self.storage_manager.delete_temp_scratch_folder()
        models_dicts = [elem[0] for elem in top_k_results]
        sorted_model_dicts = []
        for model_dict in models_dicts:
            temp_dict = [(key, value) for key, value in model_dict.items()]
            temp_dict.sort(key=lambda x: x[0])
            sorted_model_dicts.append(dict(temp_dict))

        self.storage_manager.delete_scratch_folders()

        return
