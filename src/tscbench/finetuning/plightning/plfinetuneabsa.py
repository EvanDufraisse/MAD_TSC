# -*- coding: utf-8 -*-
""" Pytorch Lighning Model Class

description

@Author: Evan Dufraisse
@Date: Fri Nov 25 2022
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2022 CEA - LASTI
"""

import pytorch_lightning as pl
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import logging
import torch
from tscbench.utils.models import ModelManager
from tscbench.modeling.models.absa.absa_model import get_model_from_config
from torchmetrics import Accuracy, F1Score
from torch.optim import AdamW
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from tscbench.modeling.models.absa.absa_model import get_token_ids_words_init
from tscbench.data.load.absa import (
    extract_all_data_from_newsmtsc_format_entry,
    AbsaDatasetLoader,
    AbsaDatasetConstraintsFiltering,
    AbsaDataCollator,
)
from tscbench.finetuning.absa.constants import (
    ABSA_MODELS_CONFIGURATORS,
    ABSA_MODELS_BUILDER,
    MODE_MASK,
)


def freeze_layers_model_encoder(model_core, layers_to_freeze, unfreeze=False):
    """
    Freeze or unfreeze layers of the model encoder

    This basically freeze the whole core model except its embeddings, which can be useful for soft-prompting.

    Args:
        model_core (torch.nn.Module): model to freeze or unfreeze
        layers_to_freeze (list[int]): list of layers to freeze or unfreeze
        unfreeze (bool, optional): unfreeze layers. Defaults to False.
    """
    if layers_to_freeze == "all":
        for param in model_core.encoder.parameters():
            param.requires_grad = unfreeze
    else:
        for layer in layers_to_freeze:
            for param in model_core.encoder.layer[layer].parameters():
                param.requires_grad = unfreeze


def freeze_embeddings_model(model_core, unfreeze=False):
    """Freeze or unfreeze embeddings of the model

    Embeddings are the translation layer between the input ids and their representative vectors.

    Args:
        model_core (torch.nn.Module): model to freeze or unfreeze
        unfreeze (bool, optional): unfreeze embeddings. Defaults to False.
    """
    for param in model_core.embeddings.parameters():
        param.requires_grad = unfreeze


def freeze_representation_layer(model, unfreeze=False):
    """
    Freeze or unfreeze representation layer of the model.

    The representation layer can be for example the MLM head for BERT, it's the intermediate transformation between the fusion layer and the classification layer.

    """
    for param in model.representation_layer.parameters():
        param.requires_grad = unfreeze


def reset_gradient_partial_wordsembeddings(
    model_core, indices_tokens=None, n_prompt_tokens=None
):
    """Reset gradient of some word embeddings of the model

    Args:
        model_core (torch.nn.Module): model to reset gradient
        indices_tokens (_type_, optional): _description_. Defaults to None.
        n_prompt_tokens (_type_, optional): _description_. Defaults to None.
    """

    second_dim = model_core.embeddings.word_embeddings.weight.shape[1]

    if not (indices_tokens is None):
        backup_grad_indices = model_core.embeddings.word_embeddings.weight.grad[
            indices_tokens
        ].clone()
        model_core.embeddings.word_embeddings.weight.grad = torch.zeros(
            model_core.embeddings.word_embeddings.weight.grad.shape,
            device=model_core.embeddings.word_embeddings.weight.device,
        )
        model_core.embeddings.word_embeddings.weight.grad[
            indices_tokens
        ] = backup_grad_indices
    else:
        model_core.embeddings.word_embeddings.weight.grad[
            : model_core.embeddings.word_embeddings.weight.shape[0] - n_prompt_tokens
        ] = torch.zeros(
            (
                model_core.embeddings.word_embeddings.weight.shape[0] - n_prompt_tokens,
                second_dim,
            ),
            device=model_core.embeddings.word_embeddings.weight.device,
        )


class PlFineTuneAbsaModel(pl.LightningModule):
    def __init__(
        self,
        model_path,
        tokenizer_path,
        sampled_config,
        gpu_pl_config,
        dataset_config,
        optimizer_config,
        model_config,
        path_ckpt=None,
        other_args={},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.sampled_config = sampled_config
        self.gpu_pl_config = gpu_pl_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.other_args = other_args

        # Load core mlm model
        self._load_model_and_tokenizer(path_ckpt)

        # Load model from state dict if ckpt supplied
        success_loading = False
        try:
            self._load_ckpt()
            success_loading = True
        except:
            print("Could not manage to load ckpt before intialization")

        # Build the absa model
        self._build_absa_model()
        if not (success_loading):
            print("Trying to load ckpt after initialization")
            self._load_ckpt()

        # Load fn dataset
        self._load_dataset_parameters()

        self._extract_hyperparameters_and_optimizer_parameters_from_configuration_files()

        self._set_validation_tracking_metrics()

    def _set_validation_tracking_metrics(self):
        self.validation_tracking_metrics = {}
        for dataset_key in self.key_dataset:
            self.validation_tracking_metrics[dataset_key] = torch.nn.ModuleDict(
                {
                    "validation_acc": Accuracy(task="multiclass", num_classes=3),
                    "validation_f1score": F1Score(
                        task="multiclass", num_classes=3, average="macro"
                    ),
                    "validation_pn_f1score": F1Score(
                        task="multiclass", num_classes=3, average="macro"
                    ),
                    "validation_pn_acc": Accuracy(task="multiclass", num_classes=3),
                }
            )
        self.validation_tracking_metrics = torch.nn.ModuleDict(
            self.validation_tracking_metrics
        )

    def _extract_hyperparameters_and_optimizer_parameters_from_configuration_files(
        self,
    ):
        self.learning_rate = self.sampled_config["lr"]
        self.seed = self.sampled_config["seeds"]
        self.effective_batch_size = self.sampled_config["batch_size"]

        self.scheduler = self.optimizer_config["optimizer"]["scheduler"]
        self.weight_decay = self.optimizer_config["optimizer"]["weight_decay"]
        self.adam_epsilon = self.optimizer_config["optimizer"]["adam_epsilon"]
        self.batch_size = self.optimizer_config["optimizer"]["batch_size_dataloader"]

        if "proportion_warmup" in self.optimizer_config["optimizer"]:
            self.proportion_warmup = self.optimizer_config["optimizer"][
                "proportion_warmup"
            ]
        else:
            self.proportion_warmup = None

        if "steps_warmup" in self.optimizer_config["optimizer"]:
            self.steps_warmup = self.optimizer_config["optimizer"]["steps_warmup"]
        else:
            self.steps_warmup = None

    def _load_dataset_parameters(self):
        self.dataset_temp_root_folder = self.dataset_config["dataset_temp_root_folder"]
        if "rw" in self.dataset_config and "mt" in self.dataset_config:
            self.multi_dataset = True
            self.name_dataset = []
            self.key_dataset = []
            for name_dataset, d_config in self.dataset_config.items():
                if name_dataset not in ["rw", "mt"]:
                    continue
                self.key_dataset.append(name_dataset)
                self.name_dataset.append(d_config["name_dataset"])
        else:
            self.multi_dataset = False
            self.name_dataset = [self.dataset_config["name_dataset"]]
            self.key_dataset = ["sg"]

    def _build_absa_model(self):
        self.absa_model_layers = get_model_from_config(
            model_mlm=self.model_mlm, tokenizer=self.tokenizer, config=self.model_config
        )
        self.model = ABSA_MODELS_BUILDER[self.model_config["absa_model"]](
            **self.absa_model_layers, **self.model_config["other_args"]
        )
        self.tokenizer = self.model.tokenizer

        logging.info("Absa model built")

    def _load_ckpt(self):
        """Load model from ckpt if path_ckpt is not None"""
        if not (self.path_ckpt is None):
            model_manager = ModelManager()
            state_dict = torch.load(self.path_ckpt, map_location=self.device).get(
                "state_dict"
            )
            self.model_mlm = model_manager.load_model_from_state_dict(
                self.model_mlm, state_dict
            )
            del state_dict
        logging.info("Building Absa model")

    def _load_model_and_tokenizer(self, path_ckpt):
        """Load model and tokenizer from path"""
        logging.info("Loading core model")
        self.model_mlm = AutoModelForMaskedLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.path_ckpt = path_ckpt
        logging.info("core model loaded")

    def setup(self, stage=None):
        # Check for each freeze possibility and freeze the model
        self._freeze_layers_core_model()
        self._freeze_embeddings_core_model()
        self._freeze_representation_layer_absa_model()

        self.n_prompt_tokens = None
        self.indices_tokens = None
        # Check for partial freeze of words embeddings, useful for softprompting
        self._partial_freeze_words_embeddings()

        # Load dataset and data_collator
        self._load_dataset_and_data_collator()

        self.total_steps = (
            len(self.datasets[self.key_dataset[0]]["train"])
            / self.sampled_config["batch_size"]
            * self.sampled_config["epochs"]
        )

        # Build the loss, if the loss is balanced it needs to have the train_dataset loaded beforehand
        self._loss_initialisation()

    def _loss_initialisation(self):
        """Initialise the loss function"""
        self.loss = self.model.loss_layer

        if hasattr(self.loss, "embs"):
            print(
                f"Before assignement device embs is {self.loss.embs.weight.get_device()}"
            )
            self.loss.embs = self.loss.embs.cuda()
            print(
                f"After assignement device embs is {self.loss.embs.weight.get_device()}"
            )
        if not (self.loss is None):
            self.loss.setup(self.datasets[self.key_dataset[0]]["train"])

    def _load_dataset_and_data_collator(self):
        """Load the dataset and the data collator"""
        model_processor = [self.model.processor]

        # Load the dataset, removing entries not fitting in the max length
        self.dataset_filtered = {}
        self.dataset_loader = {}
        self.datasets = {}
        for idx, key_dataset in enumerate(self.key_dataset):
            self.model.processor.set_return_tensors(False)
            if self.multi_dataset:
                d_config = self.dataset_config[key_dataset]
            else:
                d_config = self.dataset_config

            constraint_filter = AbsaDatasetConstraintsFiltering(
                self.dataset_temp_root_folder,
                self.name_dataset[idx],
                models_processors=model_processor,
                seed=self.seed,
                split_before_filtering=True,
            )
            self.dataset_filtered[key_dataset] = constraint_filter.constraint_filtering(
                d_config
            )

            # Change the status of the model entry processing to return torch tensors
            self.model.processor.set_return_tensors(True)

            # Load the dataset and preprocess it using the model processor
            self.dataset_loader[key_dataset] = AbsaDatasetLoader(
                self.dataset_filtered[key_dataset], model_processor[0]
            )
            self.datasets[key_dataset] = self.dataset_loader[key_dataset].load_data()
            for key, dataset in self.datasets[key_dataset].items():
                print(f"dataset {key_dataset} {key} has {len(dataset)} examples")

        # Load the data collator
        # MODE_MASK returns whether the models use a mask prompt or not
        self.data_collator = AbsaDataCollator(
            mode_mask=MODE_MASK[self.model_config["absa_model"]],
            tokenizer_mask_id=self.tokenizer.mask_token_id,
            tokenizer_padding_id=self.tokenizer.pad_token_id,
        )

    def _partial_freeze_words_embeddings(self):
        """
        Freeze the embeddings of the words in the list of words to freeze

        As we cannot directly freeze the embeddings, we need to set the gradient to 0.

        This is done at the on_before_optimizer_step call.

        """
        if (
            "partial_gradient_wordsembeddings" in self.optimizer_config
            and self.optimizer_config["partial_gradient_wordsembeddings"]
        ):
            logging.critical("partial gradient wordsembeddings")

            # If the model is a softprompt model
            if (
                self.model_config["absa_model"] == "spm"
                or self.model_config["absa_model"] == "sc"
            ):
                # Get the number of softprompt tokens
                n_prompt_tokens = self.model_config["other_args"]["n_prompt_tokens"]
                self.indices_tokens = torch.LongTensor(
                    [
                        self.tokenizer.mask_token_id,
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id,
                    ]
                    + [
                        i
                        for i in range(
                            len(self.tokenizer.vocab) - n_prompt_tokens,
                            len(self.tokenizer.vocab),
                        )
                    ]
                )
                # self.indices_tokens = []
            else:
                if self.model_config["classification_layer"]["type"] == "categorical":
                    kwargs_cl = self.model_config["classification_layer"]["kwargs"]
                    if kwargs_cl["tied"]:
                        if "words_init" in kwargs_cl:
                            self.indices_tokens = get_token_ids_words_init(
                                self.tokenizer, kwargs_cl["words_init"]
                            )
                        else:
                            self.indices_tokens = get_token_ids_words_init(
                                self.tokenizer, ["bad", "ok", "good"]
                            )

    def _freeze_representation_layer_absa_model(self):
        """
        Freeze the representation layer of the absa model

        For instance the MLM layer.

        """
        if (
            "freeze_representation_layer" in self.optimizer_config
            and self.optimizer_config["freeze_representation_layer"]
        ):
            if self.model_config["representation_layer"]["type"] != "mlm_model":
                raise ValueError(
                    "freeze_representation_layer only works with mlm_model"
                )
            freeze_representation_layer(self.model)
            logging.critical("representation layer frozen")

    def _freeze_embeddings_core_model(self):
        """
        Freeze the embeddings of the core model
        """
        if "freeze_embeddings" in self.optimizer_config:
            if self.optimizer_config["freeze_embeddings"]:
                freeze_embeddings_model(
                    self.model.core_model, self.optimizer_config["freeze_embeddings"]
                )
                logging.critical("embeddings frozen")

    def _freeze_layers_core_model(self):
        """
        Freeze the layers of the core model
        """
        if "freeze_layers" in self.optimizer_config:
            freeze_layers_model_encoder(
                self.model.core_model, self.optimizer_config["freeze_layers"]
            )
            logging.critical("layers frozen")

    def train_dataloader(self):
        if (
            "reload_dataloaders_every_n_epochs"
            in self.gpu_pl_config["pytorch_lightning_flags"]
        ):
            if self.gpu_pl_config["pytorch_lightning_flags"][
                "reload_dataloaders_every_n_epochs"
            ]:
                if "prob_shuffle" in self.gpu_pl_config["pytorch_lightning_flags"]:
                    dataset_train = self.dataset_loader[self.key_dataset[0]].load_data(
                        specific_split="train",
                        shuffle=True,
                        prob_shuffle=self.gpu_pl_config["pytorch_lightning_flags"][
                            "prob_shuffle"
                        ],
                    )["train"]
                else:
                    dataset_train = self.dataset_loader[self.key_dataset[0]].load_data(
                        specific_split="train", shuffle=True
                    )["train"]
            else:
                dataset_train = self.datasets[self.key_dataset[0]]["train"]
        else:
            dataset_train = self.datasets[self.key_dataset[0]]["train"]
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.optimizer_config["optimizer"]["batch_size_dataloader"],
            num_workers=self.optimizer_config["optimizer"]["num_workers"],
            collate_fn=self.data_collator,
            drop_last=True,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        validation_loaders = []
        for key_dataset in self.key_dataset:
            validation_loaders.append(
                torch.utils.data.DataLoader(
                    self.datasets[key_dataset]["validation"],
                    batch_size=self.optimizer_config["optimizer"][
                        "batch_size_dataloader"
                    ]
                    * 2,
                    num_workers=self.optimizer_config["optimizer"]["num_workers"],
                    collate_fn=self.data_collator,
                    drop_last=False,
                    shuffle=False,
                )
            )
        return validation_loaders

    def test_dataloader(self):
        test_loaders = []
        for key_dataset in self.key_dataset:
            test_loaders.append(
                torch.utils.data.DataLoader(
                    self.datasets[key_dataset]["test"],
                    batch_size=self.optimizer_config["optimizer"][
                        "batch_size_dataloader"
                    ]
                    * 2,
                    num_workers=self.optimizer_config["optimizer"]["num_workers"],
                    collate_fn=self.data_collator,
                    drop_last=False,
                    shuffle=False,
                )
            )
        return test_loaders

    def forward(
        self,
        batch_tokens,
        attention_mask,
        x_select=None,
        classifying_locations=None,
        counts=None,
        sentiments=None,
        params=None,
    ):
        return self.model(
            batch_tokens=batch_tokens,
            attention_mask=attention_mask,
            x_select=x_select,
            counts=counts,
            classifying_locations=classifying_locations,
        )

    def training_step(self, batch, batch_idx):
        h = self.forward(
            batch_tokens=batch[0],
            attention_mask=batch[1],
            x_select=batch[2],
            classifying_locations=batch[3],
            counts=batch[4],
        )
        sentiment_tensor = torch.tensor(batch[5], device=self.device)
        CEloss = self.loss.forward(h, sentiment_tensor)
        self.log(
            "train_loss",
            CEloss,
            on_step=True,
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )

        return CEloss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        h = self.forward(
            batch_tokens=batch[0],
            attention_mask=batch[1],
            x_select=batch[2],
            classifying_locations=batch[3],
            counts=batch[4],
        )
        sentiment_tensor = torch.tensor(batch[5], device=self.device)
        CEloss = self.loss.forward(h, sentiment_tensor)
        pn_indices = torch.where(sentiment_tensor != 1)[0]
        predicted_sentiments = torch.argmax(h, dim=1)

        # self.validation_acc(sentiment_tensor, predicted_sentiments)
        # self.validation_f1score(sentiment_tensor, predicted_sentiments)
        # self.validation_pn_f1score(
        #     sentiment_tensor[pn_indices], predicted_sentiments[pn_indices]
        # )
        # self.validation_pn_acc(
        #     sentiment_tensor[pn_indices], predicted_sentiments[pn_indices]
        # )
        self.validation_tracking_metrics[self.key_dataset[dataloader_idx]][
            "validation_acc"
        ](sentiment_tensor, predicted_sentiments)
        self.validation_tracking_metrics[self.key_dataset[dataloader_idx]][
            "validation_f1score"
        ](sentiment_tensor, predicted_sentiments)
        self.validation_tracking_metrics[self.key_dataset[dataloader_idx]][
            "validation_pn_f1score"
        ](sentiment_tensor[pn_indices], predicted_sentiments[pn_indices])
        self.validation_tracking_metrics[self.key_dataset[dataloader_idx]][
            "validation_pn_acc"
        ](sentiment_tensor[pn_indices], predicted_sentiments[pn_indices])

        self.log_dict(
            {
                f"validation_loss_{self.key_dataset[dataloader_idx]}": CEloss,
                f"validation_acc_{self.key_dataset[dataloader_idx]}": self.validation_tracking_metrics[
                    self.key_dataset[dataloader_idx]
                ][
                    "validation_acc"
                ],
                f"validation_f1score_{self.key_dataset[dataloader_idx]}": self.validation_tracking_metrics[
                    self.key_dataset[dataloader_idx]
                ][
                    "validation_f1score"
                ],
                f"validation_pn_f1score_{self.key_dataset[dataloader_idx]}": self.validation_tracking_metrics[
                    self.key_dataset[dataloader_idx]
                ][
                    "validation_pn_f1score"
                ],
                f"validation_pn_acc_{self.key_dataset[dataloader_idx]}": self.validation_tracking_metrics[
                    self.key_dataset[dataloader_idx]
                ][
                    "validation_pn_acc"
                ],
            },
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )
        return CEloss

    def on_before_optimizer_step(self, optimizer):
        if self.indices_tokens is not None:
            reset_gradient_partial_wordsembeddings(
                self.model.core_model, indices_tokens=self.indices_tokens
            )
        elif self.n_prompt_tokens is not None:
            reset_gradient_partial_wordsembeddings(
                self.model.core_model, n_prompt_tokens=self.n_prompt_tokens
            )
        else:
            pass

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
            betas=(0.9, 0.98),
        )

        if self.scheduler == "warmup_linear":
            if self.proportion_warmup is not None:
                num_warmup_steps = self.proportion_warmup * self.total_steps
            else:
                num_warmup_steps = self.steps_warmup_scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=self.total_steps,
            )
        else:
            return [optimizer]

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
