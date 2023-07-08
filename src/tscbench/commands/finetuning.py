# -*- coding: utf-8 -*-
""" Training command line parameters

@Author: Evan Dufraisse
@Date: Thu Apr 27 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
import click
from loguru import logger
from tscbench.finetuning.finetuner import FineTuner
import os


@click.group(name="finetune")
def cli_finetune():
    """Finetuning command line parameters"""
    pass


@cli_finetune.command(name="tsc", help="Finetune a TSC model")
@click.option(
    "-n", "--name-experiment", type=str, required=True, help="Name of the experiment"
)
@click.option(
    "-m",
    "--core-model-path",
    type=str,
    required=True,
    help="Path to the core model binary folder",
)
@click.option(
    "-t",
    "--tokenizer-path",
    type=str,
    required=True,
    help="Path to the tokenizer binary folder",
)
@click.option(
    "--tsc-model-config",
    type=str,
    required=True,
    help="Path to the tsc model config file",
)
@click.option(
    "--dataset-config",
    type=str,
    required=True,
    help="Path to the dataset config file",
)
@click.option(
    "--gpu-pl-config",
    type=str,
    required=True,
    help="Path to the pytorch lightning gpu config file",
)
@click.option(
    "--optimizer-config",
    type=str,
    required=True,
    help="Path to the optimizer config file",
)
@click.option(
    "--sub-path-final-folder",
    type=str,
    required=True,
    help="basename directory path to save the results",
)
@click.option("--keep-best-models", is_flag=True, help="Keep best models")
@click.option("--keep-all-models", is_flag=True, help="Keep all models")
def finetune_tsc(
    name_experiment,
    core_model_path,
    tokenizer_path,
    tsc_model_config,
    dataset_config,
    gpu_pl_config,
    optimizer_config,
    sub_path_final_folder,
    keep_best_models,
    keep_all_models,
):
    logger.info(
        "Launching finetuning of {} model with architecture {} over {} dataset".format(
            os.path.basename(core_model_path),
            os.path.basename(tsc_model_config),
            os.path.basename(dataset_config),
        )
    )
    finetuner = FineTuner(
        name_experiment=name_experiment,
        model_path=core_model_path,
        tokenizer_path=tokenizer_path,
        absa_config_path=tsc_model_config,
        dataset_config_path=dataset_config,
        gpu_pl_config_path=gpu_pl_config,
        optimizer_config_path=optimizer_config,
        subpath_final_folder=sub_path_final_folder,
        keep_best_models=keep_best_models,
        keep_all_models=keep_all_models,
    )
    finetuner.run()
