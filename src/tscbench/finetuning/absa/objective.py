# -*- coding: utf-8 -*-
""" Class AbsaOptunaObjective

description

@Author: Evan Dufraisse
@Date: Fri Nov 25 2022
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2022 CEA - LASTI
"""
import os
import optuna
import logging
import pytorch_lightning as pl
from tscbench.utils.func import find_highest_divisor
from names_generator import generate_name
from tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from tscbench.data.load.absa import AbsaDataCollator
from tscbench.finetuning.absa.constants import MODE_MASK
from pytorch_lightning.plugins.environments import SLURMEnvironment
from tscbench.finetuning.absa.constants import (
    TEMPLATES_BIAS_MENTIONS_NEWSMTSC,
    TEMPLATES_BIAS_MENTIONS_OBJECT,
)
import json
import shutil
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tscbench.finetuning.evaluation.biases_nouns_extractor import (
    BiasesCommonNounsExtractor,
)
from copy import deepcopy
from tqdm.auto import tqdm
from tscbench.data.load.absa import AbsaDatasetLoader, AbsaDatasetConstraintsFiltering
from itertools import combinations
from tscbench.finetuning.absa.constants import (
    TEMPLATES_NEWSMTSC,
    ENTITIES_NEWSMTSC,
)
import re


class AbsaOptunaObjective(object):
    def __init__(
        self,
        model_path,
        tokenizer_path,
        absa_config,
        dataset_config,
        gpu_pl_config,
        optimizer_config,
        subpath_final_folder,
        uuid,
        storage_manager,
        path_final_output_model,
        path_scratch_output_model,
        ckpt_path=None,
        other_args={},
        keep_best_models=False,
        keep_all_models=False,
        custom_datasets_json=None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.absa_config = absa_config
        self.dataset_config = dataset_config
        self.gpu_pl_config = gpu_pl_config
        self.optimizer_config = optimizer_config
        self.subpath_final_folder = subpath_final_folder
        self.ckpt_path = ckpt_path
        self.other_args = other_args
        self.uuid = uuid
        self.storage_manager = storage_manager
        self.path_final_output_model = path_final_output_model
        self.path_scratch_output_model = path_scratch_output_model
        self.keep_best_models = keep_best_models
        self.keep_all_models = keep_all_models
        self.custom_datasets_json = custom_datasets_json

    def __call__(self, trial: "optuna.trial.Trial") -> float:
        """Launch the call of the objective function. And register its outputs with the supplied optuna trial object."""

        # Register slurm characteristics of node (if any) and job id (if any) for this trial in optuna db
        node, job_id = self._register_slurm_characteristics_run_in_optuna_trial(trial)

        # Sample hyperparameters to be used for this trial
        sampled_config, hyperparameters = self._sampling_trial_hyperparameters(trial)

        print("Sampled config: {}".format(sampled_config))

        # Set seed for all packages relying on random using general pytorch_lightning function (numpy, pytorch, ...)
        print("Setting seed to {}".format(sampled_config["seeds"]))
        pl.seed_everything(sampled_config["seeds"])

        effective_batch_size, batch_size = self._determine_batch_gradient_accumulation(
            sampled_config
        )
        print("Effective batch size: {}".format(effective_batch_size))
        print("Dataloader batch size: {}".format(batch_size))

        # Load model
        pl_model = PlFineTuneAbsaModel(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            sampled_config=sampled_config,
            gpu_pl_config=self.gpu_pl_config,
            dataset_config=self.dataset_config,
            optimizer_config=self.optimizer_config,
            model_config=self.absa_config,
            path_ckpt=self.ckpt_path,
            other_args=self.other_args,
        )

        # Generate a custom name for the current trial

        custom_name = generate_name(seed=sampled_config["seeds"] + trial.number)
        print("Custom name: {}".format(custom_name))
        trial_number = trial.number

        # Get optuna metrics to return and checkpoint metrics to observe on the validation set
        optuna_metrics = self.optimizer_config["optimizer"]["optuna_metrics"]
        checkpoint_metrics = self.optimizer_config["optimizer"]["checkpoint_metrics"]

        print(f'Absa model is {self.absa_config["absa_model"]}')

        if self.absa_config["absa_model"] != "zs":
            print("Starting training")
            best_model_stats = self._train_and_evaluate_model(
                node,
                job_id,
                sampled_config,
                effective_batch_size,
                batch_size,
                pl_model,
                custom_name,
                trial_number,
                checkpoint_metrics,
            )
        else:
            # TODO check if this is correct
            best_model_stats = self._do_not_train_zero_shot(
                pl_model, custom_name, checkpoint_metrics
            )

        # Compute test set metrics
        print("Computing test and validation sets metrics")
        pl_model, results = self._compute_test_and_validation_metrics(
            pl_model, best_model_stats
        )
        # Deactivate computation of internal biases for now
        # results_internal_biases = self._compute_internal_biases_model(
        #     pl_model, best_model_stats
        # )

        if self.custom_datasets_json is not None:
            custom_datasets = json.load(open(self.custom_datasets_json, "r"))
            for dataset_name, dataset_path in custom_datasets.items():
                try:
                    dataset_path = os.path.join(
                        os.environ["DATA_SCRATCH_ABSA"].rstrip("/"),
                        self.uuid,
                        dataset_path.split("/")[-2],
                        dataset_path.split("/")[-1],
                    )
                    print(f"Computing metrics on {dataset_name}")
                    results_custom_dataset = (
                        self._compute_test_and_validation_metrics_custom_dataset(
                            pl_model, best_model_stats, dataset_path
                        )
                    )
                    results[dataset_name] = results_custom_dataset
                except Exception as e:
                    print("Error custom dataset {}".format(dataset_name))
                    print(e)

        # results_strategic_biases = {}
        # if "mt" in self.dataset_config and "rw" in self.dataset_config:
        #     results_strategic_biases = self._compute_strategic_biases_model(
        #         pl_model, best_model_stats
        #     )
        # results["strategic_biases"] = results_strategic_biases
        # results["internal_biases"] = results_internal_biases

        print("Registering configuration in results json")
        self._register_model_parameters_into_results(
            node, job_id, sampled_config, pl_model, results
        )

        print("Save json")
        self._dump_results_to_json(custom_name, results)

        print("Transfer data to final folder")
        self.storage_manager.transfer_scratch_to_final()

        print("Gather metrics to return to optuna")
        # TODO: check if this is correct
        to_return = self.extract_metrics_to_return_to_optuna(optuna_metrics, results)

        print("End of trial")
        return tuple(to_return)

    def extract_metrics_to_return_to_optuna(self, optuna_metrics, results):
        to_return = [0, 0]
        # for metric, _ in optuna_metrics.items():
        #     try:
        #         to_return.append(results[f"validation_loss_{metric}"])
        #     except:
        #         print(results.keys())
        #         raise Exception(f"Error could not find metric validation_loss_{metric}")
        # if self.absa_config["absa_model"] != "zs":
        #     shutil.rmtree(
        #         os.path.join(
        #             self.storage_manager.get_path(
        #                 self.gpu_pl_config["pytorch_lightning_params"][
        #                     "path_checkpoints"
        #                 ],
        #                 "etemp",
        #             )
        #         )
        #   )

        return to_return

    def _dump_results_to_json(self, custom_name, results):
        if not (os.path.exists(os.path.join(self.path_final_output_model))):
            os.makedirs(os.path.join(self.path_final_output_model))
        try:
            json.dump(
                results,
                open(
                    os.path.join(
                        self.path_final_output_model,
                        f'{custom_name}_val_f1_{results["validation_loss_test_f1score"]}_results.json',
                    ),
                    "w",
                ),
            )
        except:
            print(results.keys())
            json.dump(
                results,
                open(
                    os.path.join(
                        self.path_final_output_model,
                        f"{custom_name}_val_f1_{np.random.randint(1000,10000)}_results.json",
                    ),
                    "w",
                ),
            )

    def _register_model_parameters_into_results(
        self, node, job_id, sampled_config, pl_model, results
    ):
        results["seed"] = pl_model.seed
        results["sample_config"] = sampled_config
        results["optimizer_config"] = self.optimizer_config
        results["model_config"] = self.absa_config
        results["gpu_pl_config"] = self.gpu_pl_config
        results["node"] = node
        results["job_id"] = job_id
        model_hyper_identifier = sampled_config.copy()
        # if "seeds" in model_hyper_identifier:
        #     del model_hyper_identifier["seeds"]
        if "models" in model_hyper_identifier:
            del model_hyper_identifier["models"]
        key_values = [(key, value) for key, value in model_hyper_identifier.items()]
        key_values.sort(key=lambda x: x[0])
        model_hyper_identifier = "_".join(
            [f"{key}_{value}" for key, value in key_values]
        )
        model_identification = self.absa_config
        results["model_identification"] = model_identification
        results["model_hyper_identifier"] = model_hyper_identifier

    def _compute_strategic_biases_model(self, pl_model, best_model_stats):
        dataset = get_bias_newsmtsc_dataset(
            TEMPLATES_NEWSMTSC, ENTITIES_NEWSMTSC, pl_model
        )["train"]
        data_collator = AbsaDataCollator(
            mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
            tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
            tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
            force_gpu=True if torch.cuda.is_available() else False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,
            collate_fn=data_collator,
            drop_last=False,
            shuffle=False,
        )
        results_strategic_biases = {}
        for key, values in best_model_stats.items():
            predictions = []
            results_strategic_biases[key] = {}
            if self.absa_config["absa_model"] == "zs":
                continue
            else:
                pl_model = pl_model.load_from_checkpoint(values["path"])
                pl_model.setup()
            pl_model = pl_model.eval()
            pl_model = pl_model.to("cuda")
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    output = pl_model.forward(*batch)
                    out = output.cpu().tolist()
                    predictions += out
            results_strategic_biases[key] = predictions
        return results_strategic_biases

        # return results_biases

    def _compute_internal_biases_model(self, pl_model, best_model_stats):
        # Initialize dataset bias extractor
        extractor = BiasesCommonNounsExtractor(
            dataset_entries=pl_model.dataset_filtered[pl_model.key_dataset[0]]["train"]
        )

        # Extract training set biases
        extractor.extract_biases()

        # Keep only the entities that appear at least 10 times in the training set
        most_occuring_mentions = extractor.return_sorted_by_occurences(min_occ=10)

        results_biases = {}

        bias_dataset = self._build_entity_bias_dataset(
            pl_model,
            pl_model.tokenizer,
            most_occuring_mentions,
        )

        # For each checkpoint model, compute the bias metrics
        for key, values in best_model_stats.items():
            results_biases[key] = {}
            if self.absa_config["absa_model"] == "zs":
                continue
            else:
                pl_model = pl_model.load_from_checkpoint(values["path"])
                pl_model.setup()
            pl_model = pl_model.eval()
            pl_model = pl_model.to("cuda")
            results_bias_entities = self._make_inference_over_entity_bias_dataset(
                pl_model, pl_model.tokenizer, bias_dataset, most_occuring_mentions
            )
            results_biases[key]["bias_entities"] = results_bias_entities

        return results_biases

    def _return_bias_evaluation_mention_newsmtsc_format_entry(self, mention, tokenizer):
        return {
            "primary_gid": "2157",
            "sentence_normalized": " " + mention,
            "targets": [
                {
                    "Input.gid": "-1",
                    "mention": mention,
                    "polarity": 4,
                    "from": 1,
                    "to": len(mention) + 1,
                }
            ],
            # 'max_len_tokens': 50,
            # 'mentions_pos': [(1, len(mention)+1)],
            # 'all_mentions': [mention],
            # 'main_mention': mention,
            # 'main_mention_pos': (1, len(mention)+1),
            # 'sentiment': 4,
            # 'processed_mention_tokens': [(1, len(tokenizer.encode(" "+mention, add_special_tokens=False)))],
            # 'processed_mentions_pos': [(1, len(mention)+1)]
        }

    def _make_inference_over_entity_bias_dataset(
        self, pl_model, tokenizer, bias_dataset, mentions, batch_size=16
    ):
        data_collator = AbsaDataCollator(
            mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
            tokenizer_mask_id=tokenizer.mask_token_id,
            tokenizer_padding_id=tokenizer.pad_token_id,
            force_gpu=True if torch.cuda.is_available() else False,
        )
        bias_dataloader = torch.utils.data.DataLoader(
            bias_dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=data_collator,
            drop_last=False,
            shuffle=False,
        )
        results = {}
        predictions = []
        with torch.no_grad():
            for batch in tqdm(bias_dataloader):
                output = pl_model.forward(*batch)
                out = output.cpu().tolist()
                predictions += out
        for i, mention in enumerate(mentions):
            results[mention["mention"]] = (i, mention, predictions[i])
        return results

    def _build_entity_bias_dataset(self, pl_model, tokenizer, mentions):
        mention_dataset = {
            "train": [
                self._return_bias_evaluation_mention_newsmtsc_format_entry(
                    mention["mention"], tokenizer
                )
                for mention in mentions
            ]
        }
        dataset_loader = AbsaDatasetLoader(mention_dataset, pl_model.model.processor)
        datasets = dataset_loader.load_data()
        return datasets["train"]

    def _compute_test_and_validation_metrics(self, pl_model, best_model_stats):
        dataloaders = {
            "test": pl_model.test_dataloader,
            "validation": pl_model.val_dataloader,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = {}
        data_collator = AbsaDataCollator(
            mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
            tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
            tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
            force_gpu=True if torch.cuda.is_available() else False,
        )
        for name_dataloader, _ in dataloaders.items():
            print(name_dataloader)
            for idx_dataloader, key_dataset in enumerate(pl_model.key_dataset):
                # dataloader = dataloaders[idx_dataloader]
                print(key_dataset)
                for key, values in best_model_stats.items():
                    predictions = []
                    true_labels = []

                    print("###")
                    print(key)
                    # print(values["path"])
                    print("###")
                    if self.absa_config["absa_model"] == "zs":
                        pass
                    else:
                        pl_model = pl_model.load_from_checkpoint(values["path"])
                        pl_model.setup()
                    pl_model = pl_model.eval()
                    pl_model = pl_model.to(device)
                    if hasattr(pl_model.loss, "embs"):
                        print(
                            f"Before assignement device embs is {pl_model.loss.embs.weight.get_device()}"
                        )
                        pl_model.loss.embs = pl_model.loss.embs.cpu()
                    if (
                        hasattr(pl_model.loss.loss_layer, "weight")
                        and pl_model.loss.loss_layer.weight is not None
                    ):
                        print(
                            f"Before assignement device loss_layer is {pl_model.loss.loss_layer.weight.get_device()}"
                        )
                        pl_model.loss.loss_layer.weight = (
                            pl_model.loss.loss_layer.weight.cpu()
                        )
                    results[f"{key}_epoch"] = int(pl_model.current_epoch)
                    results[f"{key}_global_step"] = int(pl_model.global_step)

                    if name_dataloader == "test":
                        print("loading test dataloader")
                        dataloader = torch.utils.data.DataLoader(
                            pl_model.datasets[key_dataset]["test"],
                            batch_size=pl_model.optimizer_config["optimizer"][
                                "batch_size_dataloader"
                            ]
                            * 2,
                            num_workers=0,
                            collate_fn=data_collator,
                            drop_last=False,
                            shuffle=False,
                        )
                    else:
                        print("loading val dataloader")
                        dataloader = torch.utils.data.DataLoader(
                            pl_model.datasets[key_dataset]["validation"],
                            batch_size=pl_model.optimizer_config["optimizer"][
                                "batch_size_dataloader"
                            ]
                            * 2,
                            num_workers=0,
                            collate_fn=data_collator,
                            drop_last=False,
                            shuffle=False,
                        )
                    iter_data = iter(dataloader)
                    print(f"start predictions for {name_dataloader}")
                    with torch.no_grad():
                        for (
                            batch_tokens,
                            attention_mask,
                            x_select,
                            classifying_locations,
                            counts,
                            sentiments,
                            params,
                        ) in tqdm(iter_data):
                            out = pl_model.forward(
                                batch_tokens=batch_tokens,
                                attention_mask=attention_mask,
                                classifying_locations=classifying_locations,
                                counts=counts,
                                x_select=x_select,
                            )
                            true_labels += list(sentiments.tolist())
                            a = out.detach().cpu().tolist()
                            if type(a) == int:
                                raise Exception("Error int instead of tensor")
                                predictions.append(a)
                            else:
                                predictions += list(a)
                        print("predictions done")
                        print(len(predictions))
                    # print(predictions.shape)
                    # print(true_labels.shape)
                    # print(f"{key}_{name_dataloader}_predictions")
                    # print(len(predictions))
                    # print(len(true_labels))
                    results[
                        f"{key_dataset}_{key}_{name_dataloader}_predictions"
                    ] = predictions
                    results[
                        f"{key_dataset}_{key}_{name_dataloader}_true_labels"
                    ] = true_labels
                    # print(results.keys())

                    print("computing metrics predictions...")
                    print(pl_model.loss.loss_layer)
                    print(hasattr(pl_model.loss.loss_layer, "weight"))
                    print(dir(pl_model.loss.loss_layer))
                    try:
                        predictions_tensor = torch.FloatTensor(predictions)
                        true_labels_tensor = torch.LongTensor(true_labels)
                    except:
                        print(predictions)
                        print(true_labels)
                        raise Exception("Error converting to tensor")
                    pn_indices = torch.where(true_labels_tensor != 1)[0]
                    if hasattr(pl_model.loss, "embs"):
                        print(
                            f"Before assignement device embs is {pl_model.loss.embs.weight.get_device()}"
                        )
                        pl_model.loss.embs = pl_model.loss.embs.cpu()
                    if (
                        hasattr(pl_model.loss.loss_layer, "weight")
                        and pl_model.loss.loss_layer.weight is not None
                    ):
                        print(
                            f"Before assignement device loss_layer is {pl_model.loss.loss_layer.weight.get_device()}"
                        )
                        pl_model.loss.loss_layer.weight = (
                            pl_model.loss.loss_layer.weight.cpu()
                        )
                    results[f"{key_dataset}_{key}_{name_dataloader}_loss"] = float(
                        pl_model.model.loss_layer.forward(
                            predictions_tensor, true_labels_tensor
                        ).squeeze()
                    )
                    predictions_numpy = np.argmax(
                        predictions_tensor.cpu().detach().numpy(), axis=-1
                    )
                    true_labels_numpy = true_labels_tensor.cpu().detach().numpy()
                    pn_indices = pn_indices.cpu().detach().numpy()
                    results[f"{key_dataset}_{key}_{name_dataloader}_f1score"] = float(
                        f1_score(true_labels_numpy, predictions_numpy, average="macro")
                    )
                    results[f"{key_dataset}_{key}_{name_dataloader}_acc"] = float(
                        accuracy_score(true_labels_numpy, predictions_numpy)
                    )
                    results[f"{key_dataset}_{key}_{name_dataloader}_all_f1score"] = [
                        float(elem)
                        for elem in f1_score(
                            true_labels_numpy,
                            predictions_numpy,
                            average=None,
                        )
                    ]
                    results[f"{key_dataset}_{key}_{name_dataloader}_pn_acc"] = float(
                        accuracy_score(
                            true_labels_numpy[pn_indices], predictions_numpy[pn_indices]
                        )
                    )
                    print("computing metrics predictions done")

        return pl_model, results

    def load_custom_datasets(self, pl_model, mode_mask, tokenizer, path_dataset):
        data_collator = AbsaDataCollator(
            mode_mask=mode_mask,
            tokenizer_mask_id=tokenizer.mask_token_id,
            tokenizer_padding_id=tokenizer.pad_token_id,
            force_gpu=True if torch.cuda.is_available() else False,
        )
        loaded_datasets = {}
        pl_model.model.processor.set_return_tensors(False)
        model_processor = [pl_model.model.processor]
        constraint_filter = AbsaDatasetConstraintsFiltering(
            name_dataset="dataset",
            root_folder=None,
            path_dataset=path_dataset,
            models_processors=model_processor,
        )
        dataset_filtered = constraint_filter.constraint_filtering()
        pl_model.model.processor.set_return_tensors(True)
        dataset_loader = AbsaDatasetLoader(dataset_filtered, model_processor[0])
        loaded_dataset = dataset_loader.load_data()
        loaded_dataset = list(loaded_dataset.values())[0]
        pl_model.model.processor.set_return_tensors(False)
        return loaded_dataset, data_collator

    def _compute_test_and_validation_metrics_custom_dataset(
        self, pl_model, best_model_stats, path_dataset
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = {}
        mode_mask = MODE_MASK[pl_model.model_config["absa_model"]]
        tokenizer = pl_model.tokenizer
        loaded_dataset, data_collator = self.load_custom_datasets(
            pl_model, mode_mask, tokenizer, path_dataset
        )
        name_dataloader = "_".join(path_dataset.split("/")[-2:]).replace(".", "")
        for key, values in best_model_stats.items():
            predictions = []
            true_labels = []

            print("###")
            print(key)
            # print(values["path"])
            print("###")
            if self.absa_config["absa_model"] == "zs":
                pass
            else:
                pl_model = pl_model.load_from_checkpoint(values["path"])
                pl_model.setup()
            pl_model = pl_model.eval()
            pl_model = pl_model.to(device)
            if hasattr(pl_model.loss, "embs"):
                print(
                    f"Before assignement device embs is {pl_model.loss.embs.weight.get_device()}"
                )
                pl_model.loss.embs = pl_model.loss.embs.cpu()
            if (
                hasattr(pl_model.loss.loss_layer, "weight")
                and pl_model.loss.loss_layer.weight is not None
            ):
                print(
                    f"Before assignement device loss_layer is {pl_model.loss.loss_layer.weight.get_device()}"
                )
                pl_model.loss.loss_layer.weight = pl_model.loss.loss_layer.weight.cpu()
            results[f"{key}_epoch"] = int(pl_model.current_epoch)
            results[f"{key}_global_step"] = int(pl_model.global_step)

            dataloader = torch.utils.data.DataLoader(
                loaded_dataset,
                batch_size=pl_model.optimizer_config["optimizer"][
                    "batch_size_dataloader"
                ]
                * 2,
                num_workers=0,
                collate_fn=data_collator,
                drop_last=False,
                shuffle=False,
            )
            iter_data = iter(dataloader)
            print(f"start predictions for {name_dataloader}")
            with torch.no_grad():
                for (
                    batch_tokens,
                    attention_mask,
                    x_select,
                    classifying_locations,
                    counts,
                    sentiments,
                    params,
                ) in tqdm(iter_data):
                    out = pl_model.forward(
                        batch_tokens=batch_tokens,
                        attention_mask=attention_mask,
                        classifying_locations=classifying_locations,
                        counts=counts,
                        x_select=x_select,
                    )
                    true_labels += list(sentiments.tolist())
                    a = out.detach().cpu().tolist()
                    if type(a) == int:
                        raise Exception("Error int instead of tensor")
                        predictions.append(a)
                    else:
                        predictions += list(a)
                print("predictions done")
                print(len(predictions))
            results[f"custom_{key}_{name_dataloader}_predictions"] = predictions
            results[f"custom_{key}_{name_dataloader}_true_labels"] = true_labels

            print("computing metrics predictions...")
            print(pl_model.loss.loss_layer)
            print(hasattr(pl_model.loss.loss_layer, "weight"))
            print(dir(pl_model.loss.loss_layer))
            try:
                predictions_tensor = torch.FloatTensor(predictions)
                true_labels_tensor = torch.LongTensor(true_labels)
            except:
                print(predictions)
                print(true_labels)
                raise Exception("Error converting to tensor")
            pn_indices = torch.where(true_labels_tensor != 1)[0]
            if hasattr(pl_model.loss, "embs"):
                print(
                    f"Before assignement device embs is {pl_model.loss.embs.weight.get_device()}"
                )
                pl_model.loss.embs = pl_model.loss.embs.cpu()
            if (
                hasattr(pl_model.loss.loss_layer, "weight")
                and pl_model.loss.loss_layer.weight is not None
            ):
                print(
                    f"Before assignement device loss_layer is {pl_model.loss.loss_layer.weight.get_device()}"
                )
                pl_model.loss.loss_layer.weight = pl_model.loss.loss_layer.weight.cpu()
            results[f"custom_{key}_{name_dataloader}_loss"] = float(
                pl_model.model.loss_layer.forward(
                    predictions_tensor, true_labels_tensor
                ).squeeze()
            )
            predictions_numpy = np.argmax(
                predictions_tensor.cpu().detach().numpy(), axis=-1
            )
            true_labels_numpy = true_labels_tensor.cpu().detach().numpy()
            pn_indices = pn_indices.cpu().detach().numpy()
            results[f"custom_{key}_{name_dataloader}_f1score"] = float(
                f1_score(true_labels_numpy, predictions_numpy, average="macro")
            )
            results[f"custom_{key}_{name_dataloader}_acc"] = float(
                accuracy_score(true_labels_numpy, predictions_numpy)
            )
            results[f"custom_{key}_{name_dataloader}_all_f1score"] = [
                float(elem)
                for elem in f1_score(
                    true_labels_numpy,
                    predictions_numpy,
                    average=None,
                )
            ]
            results[f"custom_{key}_{name_dataloader}_pn_acc"] = float(
                accuracy_score(
                    true_labels_numpy[pn_indices], predictions_numpy[pn_indices]
                )
            )
            print("computing metrics predictions done")

        return results

    def _do_not_train_zero_shot(self, pl_model, custom_name, checkpoint_metrics):
        pl_model.setup()
        best_model_stats = {
            metric: {
                "path": "No path for zs",
                "model_info": os.path.join(
                    os.path.join(
                        self.storage_manager.get_path(
                            self.gpu_pl_config["pytorch_lightning_params"][
                                "path_checkpoints"
                            ],
                            "etemp",
                        ),
                        metric,
                    ),
                    custom_name
                    + "#{monitor_key:.4f}".replace("monitor_key", metric)
                    + ".json",
                ),
                "score": 0,
            }
            for metric in checkpoint_metrics
        }
        print(best_model_stats)
        return best_model_stats

    def _train_and_evaluate_model(
        self,
        node,
        job_id,
        sampled_config,
        effective_batch_size,
        batch_size,
        pl_model,
        custom_name,
        trial_number,
        checkpoint_metrics,
    ):
        path_logging_folder = os.path.join(
            self.path_scratch_output_model,
            self.gpu_pl_config["pytorch_lightning_params"]["log_dir"],
        )
        os.makedirs(
            os.path.join(path_logging_folder, f"{custom_name}_trial_{trial_number}"),
            exist_ok=True,
        )
        logger = TensorBoardLogger(
            path_logging_folder, f"{custom_name}_trial_{trial_number}"
        )

        logging.critical(f"{os.listdir(path_logging_folder)}")
        logging.critical(f"{os.path.isdir(path_logging_folder)}")

        checkpoints_paths = {}

        print("Creating checkpoint callback")
        if "sg" in pl_model.key_dataset:
            add_suffix = False
        else:
            add_suffix = True

        for idx, key_dataset in enumerate(pl_model.key_dataset):
            for metric in checkpoint_metrics:
                metric_supp = metric + "_" + key_dataset
                if add_suffix:
                    metric_supp += f"/dataloader_idx_{idx}"

                print(
                    "Creating checkpoint callback for metric ",
                    metric_supp,
                    checkpoint_metrics[metric],
                )
                checkpoints_paths[metric_supp] = {
                    "path_checkpoint": os.path.join(
                        self.storage_manager.get_path(
                            self.gpu_pl_config["pytorch_lightning_params"][
                                "path_checkpoints"
                            ],
                            "etemp",
                        ),
                        metric_supp,
                    ),
                    "direction": checkpoint_metrics[metric],
                }

        checkpoints_callbacks = {}
        for monitor_key, checkpoint_config in checkpoints_paths.items():
            checkpoints_callbacks[monitor_key] = ModelCheckpoint(
                dirpath=checkpoint_config["path_checkpoint"],
                filename=custom_name
                + "#{monitor_key:.4f}".replace("monitor_key", monitor_key),
                save_top_k=self.gpu_pl_config["pytorch_lightning_params"][
                    "checkpoints_topk"
                ],
                verbose=True,
                monitor=monitor_key,
                mode=checkpoint_config["direction"],
            )

        callbacks = []
        if self.gpu_pl_config["pytorch_lightning_params"]["lr_monitor"]:
            print("Creating early stopping callback")
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks += [lr_monitor] + list(checkpoints_callbacks.values())

        if len(pl_model.key_dataset) == 1:
            early_stopping_validation_loss_callback = EarlyStopping(
                monitor="validation_f1score_sg", patience=30, verbose=True, mode="max"
            )
            callbacks.append(early_stopping_validation_loss_callback)

        early_stopping_training_loss_callback = EarlyStopping(
            monitor="train_loss",
            patience=1000,
            verbose=True,
            stopping_threshold=1e-7,
            check_on_train_epoch_end=True,
            mode="min",
        )

        callbacks.append(early_stopping_training_loss_callback)

        print("Loading pl configuration flags")
        pl_flags = self.gpu_pl_config["pytorch_lightning_flags"]
        pl_flags["accumulate_grad_batches"] = effective_batch_size // batch_size
        pl_flags["max_epochs"] = sampled_config["epochs"]
        print("Max epochs ", pl_flags["max_epochs"])

        trainer = pl.Trainer(
            **pl_flags,
            logger=logger,
            callbacks=callbacks,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_progress_bar=False,
        )

        print("Starting training")
        trainer.fit(pl_model)
        print("Training finished")

        # TODO
        # self.name_dataset = self.dataset_config["name_dataset"]
        # if self.name_dataset in ["laptops", "restaurants", "mam"]:
        #     TEST_BIAS = TEMPLATES_BIAS_MENTIONS_OBJECT
        # elif self.name_dataset in ["newsmtscmt", "newsmtscrw", "newsmtsc"]:
        #     TEST_BIAS = TEMPLATES_BIAS_MENTIONS_NEWSMTSC

        print("Retrieving best models checkpoints")
        best_model_stats = {
            key: {
                "path": checkpoints_callbacks[key].best_model_path,
                "model_info": checkpoints_callbacks[key].best_model_path.split(".ckpt")[
                    0
                ]
                + ".json",
                "score": checkpoints_callbacks[key].best_model_score,
            }
            for key in checkpoints_callbacks.keys()
        }

        if self.keep_best_models or self.keep_all_models:
            print("Copying best models checkpoints to final folder")
            j_characteristics_model = {}
            j_characteristics_model["seed"] = pl_model.seed
            j_characteristics_model["sample_config"] = sampled_config
            j_characteristics_model["optimizer_config"] = self.optimizer_config
            j_characteristics_model["model_config"] = self.absa_config
            j_characteristics_model["gpu_pl_config"] = self.gpu_pl_config
            j_characteristics_model["node"] = node
            j_characteristics_model["job_id"] = job_id

            for checkpoint_key, values in best_model_stats.items():
                if not (
                    os.path.exists(
                        self.storage_manager.get_path(checkpoint_key, "final")
                    )
                ):
                    os.makedirs(self.storage_manager.get_path(checkpoint_key, "final"))

                score = float(values["score"])
                available_models = [
                    file
                    for file in os.listdir(
                        self.storage_manager.get_path(checkpoint_key, "final")
                    )
                    if file.endswith(".json")
                ]
                print(available_models)
                print(self.storage_manager.get_path(checkpoint_key, "final"))
                scores = [
                    json.load(
                        open(
                            os.path.join(
                                self.storage_manager.get_path(checkpoint_key, "final"),
                                file,
                            )
                        )
                    )["score"]
                    for file in available_models
                ]
                if "validation_loss" in checkpoint_key and not (self.keep_all_models):
                    if checkpoints_paths[checkpoint_key]["direction"] == "min":
                        if len(scores) == 0 or score < min(scores):
                            files = [
                                os.path.join(
                                    self.storage_manager.get_path(
                                        checkpoint_key, "final"
                                    ),
                                    file,
                                )
                                for file in os.listdir(
                                    self.storage_manager.get_path(
                                        checkpoint_key, "final"
                                    )
                                )
                                if file.endswith(".ckpt")
                            ]
                            for file in files:
                                os.remove(file)
                            shutil.copy(
                                values["path"],
                                self.storage_manager.get_path(checkpoint_key, "final"),
                            )
                    if checkpoints_paths[checkpoint_key]["direction"] == "max":
                        if len(scores) == 0 or score > max(scores):
                            files = [
                                os.path.join(
                                    self.storage_manager.get_path(
                                        checkpoint_key, "final"
                                    ),
                                    file,
                                )
                                for file in os.listdir(
                                    self.storage_manager.get_path(
                                        checkpoint_key, "final"
                                    )
                                )
                                if file.endswith(".ckpt")
                            ]
                            for file in files:
                                os.remove(file)
                            shutil.copy(
                                values["path"],
                                self.storage_manager.get_path(checkpoint_key, "final"),
                            )
                if self.keep_all_models:
                    shutil.copy(
                        values["path"],
                        self.storage_manager.get_path(checkpoint_key, "final"),
                    )
                j_characteristics_model["score"] = score
                json.dump(
                    j_characteristics_model,
                    open(
                        os.path.join(
                            self.storage_manager.get_path(checkpoint_key, "final"),
                            f"{custom_name}_{score}.json",
                        ),
                        "w",
                    ),
                )
        return best_model_stats

    def _determine_batch_gradient_accumulation(self, sampled_config):
        """Based on the target effective batch_size and the maximum batch_size supported by the GPU, determine the number of gradient accumulation steps and return the batch_size for the dataloader.

        Args:
            sampled_config (dict): dict containing the diverse set of hyperparameters for the current trial, notably the effective batch_size.

        Returns:
            (int, int): (effective_batch_size, batch_size of dataloader)
        """
        effective_batch_size = sampled_config["batch_size"]
        print(self.optimizer_config)
        max_batch_size_per_gpu = self.optimizer_config["optimizer"][
            "max_batch_size_per_gpu"
        ]
        print(f"Max batch size per gpu is {max_batch_size_per_gpu}")
        batch_size = find_highest_divisor(
            effective_batch_size, 1, max_batch_size_per_gpu
        )
        print(
            f"Effective batch size is {effective_batch_size}, batch size per gpu is {batch_size}"
        )
        self.optimizer_config["optimizer"]["batch_size_dataloader"] = batch_size
        return effective_batch_size, batch_size

    def _sampling_trial_hyperparameters(self, trial):
        logging.info("Sampling hyperparameters")
        sampled_config = {}

        hyperparameters = self.optimizer_config["optimizer"]["hyperparameters"]
        for key, value in hyperparameters.items():
            if type(value[0]) == type(1):
                sampled_config[key] = trial.suggest_int(key, min(value), max(value))
            elif type(value[0]) == type(1.0):
                sampled_config[key] = trial.suggest_float(key, min(value), max(value))
            else:
                sampled_config[key] = trial.suggest_categorical(key, value)
        return sampled_config, hyperparameters

    def _register_slurm_characteristics_run_in_optuna_trial(self, trial):
        """Register slurm characteristics in optuna trial

        Args:
            trial (optuna.Trial): optuna trial

        Returns:
            [str,str]: node, job_id
        """
        print("Check type of computing device")
        if "SLURMD_NODENAME" in os.environ:
            node = os.environ["SLURMD_NODENAME"]
        else:
            node = "local"

        print(f"Found {node} as computing device")

        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
        else:
            job_id = 0

        print(f"job id is {job_id}")

        trial.set_user_attr("host", node)
        trial.set_user_attr("job", job_id)
        return node, job_id


def replace_one_entity(template, entity, pos_entity):
    entity_str = "{entity{pos_entity}}".replace("{pos_entity}", str(pos_entity))
    pron_str = "{pron{pos_entity}}".replace("{pos_entity}", str(pos_entity))
    poss_str = "{poss{pos_entity}}".replace("{pos_entity}", str(pos_entity))
    template["template"] = template["template"].replace(entity_str, entity["entity"])
    template["template"] = template["template"].replace(pron_str, entity["pron"])
    template["template"] = template["template"].replace(poss_str, entity["poss"])
    return template


def get_number_entities(template):
    return len(list(set(re.findall(r"{entity\d}", template["template"]))))


def create_entry(template, entity_name, entities_names, idx, idx_entry):
    positions = get_char_pos_pattern(entity_name, template["template"])
    try:
        assert len(positions) >= 1
    except:
        print(entities_names)
        print(template["template"])
    position = positions[0]
    pos_entity = idx + 1
    suffix = "_".join([f"{entities_names[i]}" for i in range(len(entities_names))])
    return {
        "primary_gid": f"{idx_entry}_entity{pos_entity}_{suffix}",
        "sentence_normalized": template["template"],
        "targets": [
            {
                "Input.gid": f"{idx}_entity{pos_entity}_{suffix}",
                "from": position[0],
                "to": position[1],
                "mention": entity_name,
                "polarity": template["sentiments"][idx],
            }
        ],
    }


def generate_newsmtsc_entry(template, entities, idx_entry):
    n_entities = get_number_entities(template)
    entities_names = []
    entries = []
    comb = list(combinations(entities, n_entities))
    for combination in comb:
        entities_names = []
        copy_template = deepcopy(template)
        for i, entity in enumerate(combination):
            copy_template = replace_one_entity(copy_template, entity, i + 1)
            entities_names.append(entity["entity"])
        for idx, entity_name in enumerate(entities_names):
            entry = create_entry(
                copy_template, entity_name, entities_names, idx, idx_entry
            )
            entries.append(entry)
    return entries


def get_char_pos_pattern(pattern, text):
    return [(m.start(), m.start() + len(pattern)) for m in re.finditer(pattern, text)]


def get_bias_newsmtsc_dataset(templates, entities, pl_model):
    entries = []
    for idx, template in enumerate(templates):
        entries += generate_newsmtsc_entry(template, entities, idx)
    pl_model.model.processor.set_return_tensors(True)

    dataset_loader = AbsaDatasetLoader({"train": entries}, pl_model.model.processor)
    dataset = dataset_loader.load_data()
    return dataset
