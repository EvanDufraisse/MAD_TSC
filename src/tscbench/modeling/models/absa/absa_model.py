#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch
from abc import abstractmethod
from tscbench.modeling.blocks.classification_layers import *
from tscbench.modeling.blocks.fusion_layers import *
from tscbench.modeling.blocks.representation_layers import *
from tscbench.modeling.blocks.loss_layers import *


# An AbsaModel must be build from:
# - a core model with MLM head (roberta, bert)
# - a tokenizer (that can be custom in the case of softprompting)
# - a fusion layer that must combine several representations into a single vector
# - a representation layer (bert, roberta, dummy), this layer is used to compute a representation using the latter vector
# - a classification layer (ordered, categorical), this layer classifies the representation into (pos, neu, neg)


def get_word_representation_model(
    model_core, tokenizer, words=["bad", "ok", "good"], tokens=None, tied=False
):
    if hasattr(model_core, "bert") or model_core.config.to_dict()["architectures"][
        0
    ].startswith("Bert"):
        try:
            word_embeddings = model_core.bert.embeddings.word_embeddings.weight
        except:
            word_embeddings = model_core.embeddings.word_embeddings.weight
    elif (
        hasattr(model_core, "roberta")
        or model_core.config.to_dict()["architectures"][0].startswith("Roberta")
        or model_core.config.to_dict()["architectures"][0].startswith("Camembert")
        or model_core.config.to_dict()["architectures"][0].startswith("XLMRoberta")
    ):
        try:
            word_embeddings = model_core.roberta.embeddings.word_embeddings.weight
        except:
            word_embeddings = model_core.embeddings.word_embeddings.weight
    out = []
    token_ids = []
    if not (tokens is None):
        for token in tokens:
            if not (tied):
                out.append(word_embeddings[token])
    else:
        for word in words:
            token_id = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
            token_ids.append(token_id)
            assert len(token_id) == 1
            if not (tied):
                out.append(word_embeddings[token_id[0]])

    token_ids = torch.LongTensor(token_ids)
    if len(token_ids.shape) > 1:
        token_ids = torch.LongTensor(token_ids).squeeze(1)
    else:
        token_ids = torch.LongTensor(token_ids)
    if not (tied):
        return torch.stack(out), token_ids
    else:
        return word_embeddings, token_ids


def get_token_ids_words_init(tokenizer, words=["bad", "ok", "good"]):
    token_ids = []
    for word in words:
        token_id = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        token_ids.append(token_id)
        assert len(token_id) == 1
    token_ids = torch.LongTensor(token_ids)
    if len(token_ids.shape) > 1:
        token_ids = torch.LongTensor(token_ids).squeeze(1)
    else:
        token_ids = torch.LongTensor(token_ids)
    return token_ids


def get_ordered_word_representation_init(
    model_core, tokenizer, good_words=["good"], bad_words=["bad"]
):
    # if model_core.config.to_dict()["architectures"][0].startswith("Bert"):
    #     try:
    #         word_embeddings = model_core.bert.embeddings.word_embeddings.weight
    #     except:
    #         word_embeddings = model_core.embeddings.word_embeddings.weight
    # elif model_core.config.to_dict()["architectures"][0].startswith("Roberta"):
    #     try:
    #         word_embeddings = model_core.roberta.embeddings.word_embeddings.weight
    #     except:
    #         word_embeddings = model_core.embeddings.word_embeddings.weight
    if hasattr(model_core, "roberta"):
        word_embeddings = model_core.roberta.embeddings.word_embeddings.weight
    elif hasattr(model_core, "bert"):
        word_embeddings = model_core.bert.embeddings.word_embeddings.weight
    elif hasattr(model_core, "embeddings"):
        word_embeddings = model_core.embeddings.word_embeddings.weight
    out = []
    token_ids = []
    class_words = {"good": good_words, "bad": bad_words}
    tensors_words = {"good": [], "bad": []}
    for class_word in ["good", "bad"]:
        for word in class_words[class_word]:
            token_id = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
            token_ids.append(token_id)
            assert len(token_id) == 1
            tensors_words[class_word].append(word_embeddings[token_id[0]])
    return (
        torch.mean(torch.stack(tensors_words["bad"]), dim=0).squeeze()
        - torch.mean(torch.stack(tensors_words["good"]), dim=0).squeeze()
    )


class AbsaModel(torch.nn.Module):
    def __init__(
        self,
        core_model,
        tokenizer,
        fusion_layer,
        representation_layer,
        classification_layer,
        loss_layer=None,
    ):
        super().__init__()
        self.core_model = core_model
        self.tokenizer = tokenizer
        self.fusion_layer = fusion_layer
        self.representation_layer = representation_layer
        self.classification_layer = classification_layer
        self.loss_layer = loss_layer

    @abstractmethod
    def data_generator(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_loss_layer(self):
        if not (self.loss_layer is None):
            return self.loss_layer
        else:
            raise Exception("No loss layer defined")

    def set_loss_layer(self, loss_layer):
        self.loss_layer = loss_layer

    def save_tokenizer(self, path):
        self.tokenizer.save_pretrained(path)


def separate_core_head_model(mlm_model):
    """
    Separate the core model from the MLM head of a model.
    """
    # if mlm_model.config.to_dict()["architectures"][0].startswith("Bert"):
    if hasattr(mlm_model, "bert") or mlm_model.config.to_dict()["architectures"][
        0
    ].startswith("Bert"):
        return mlm_model.bert, mlm_model.cls
    elif (
        hasattr(mlm_model, "roberta")
        or mlm_model.config.to_dict()["architectures"][0].startswith("Roberta")
        or mlm_model.config.to_dict()["architectures"][0].startswith("Camembert")
        or mlm_model.config.to_dict()["architectures"][0].startswith("XLMRoberta")
    ):
        return mlm_model.roberta, mlm_model.lm_head
    else:
        raise ValueError(
            f"Unknown architecture {mlm_model.config.to_dict()['architectures'][0]}"
        )


def get_model_layers(
    model_mlm,
    tokenizer,
    fusion_layer="max_pooling",
    representation_layer="none",
    classification_layer="ordered",
    loss_layer="CE",
):
    """_summary_

    Args:
        model_mlm (ModelForMaskedLM): model to use
        tokenizer (FastTokenizer): hg tokenizer
        fusion_layer (str, optional): "max_pooling", "mean", "max_pooling_abs", "attention", "cls", "select". Defaults to "max_pooling".
        representation_layer (str, optional): "mlm_config", "mlm_model", "cls_pooler", "none" . Defaults to "none".
        classification_layer (str, optional): "categorical", "ordered" . Defaults to "ordered".
        loss_layer (str, optional): "CE", "BCElogits"
    """
    core_model, mlm_head = separate_core_head_model(model_mlm)

    fusion_layer_type = fusion_layer["type"]
    fusion_layer_kwargs = fusion_layer["kwargs"]
    if fusion_layer_type == "max_pooling":
        fusion_layer = FilterFusionLayer(
            MaxPoolingFusionLayer(abs=False, **fusion_layer_kwargs)
        )
    elif fusion_layer_type == "mean":
        fusion_layer = FilterFusionLayer(MeanFusionLayer(**fusion_layer_kwargs))
    elif fusion_layer_type == "max_pooling_abs":
        fusion_layer = FilterFusionLayer(
            MaxPoolingFusionLayer(abs=True, **fusion_layer_kwargs)
        )
    elif fusion_layer_type == "attention":
        fusion_layer = FilterFusionLayer(AttentionFusionLayer(**fusion_layer_kwargs))
    elif fusion_layer_type == "cls":
        fusion_layer = ClsTokenOnly(**fusion_layer_kwargs)
    elif fusion_layer_type == "select":
        fusion_layer = SelectFusionLayer(**fusion_layer_kwargs)
    else:
        raise ValueError(f"Unknown fusion layer {fusion_layer}")

    architecture = core_model.config.to_dict()["architectures"][0]
    if hasattr(core_model, "bert") or architecture.startswith("Bert"):
        arch_name = "Bert"
        arch_mlm = BertMlmRepresentationLayer
    elif (
        hasattr(core_model, "roberta")
        or architecture.startswith("Roberta")
        or architecture.startswith("Camembert")
        or architecture.startswith("XLMRoberta")
    ):
        arch_name = "Roberta"
        arch_mlm = RobertaMlmRepresentationLayer
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    representation_layer_type = representation_layer["type"]
    representation_layer_kwargs = representation_layer["kwargs"]
    if representation_layer_type == "mlm_config":
        representation_layer = arch_mlm(
            from_config=model_mlm.config, **representation_layer_kwargs
        )
    elif representation_layer_type == "mlm_model":
        representation_layer = arch_mlm(
            from_model=model_mlm, **representation_layer_kwargs
        )
    elif representation_layer_type == "cls_pooler":
        representation_layer = ClsPooler(
            config=model_mlm.config, **representation_layer_kwargs
        )
    elif representation_layer_type == "none":
        representation_layer = DummyRepresentationLayer(**representation_layer_kwargs)

    hidden_size = core_model.config.to_dict()["hidden_size"]
    classification_layer_type = classification_layer["type"]
    classification_layer_kwargs = classification_layer["kwargs"]
    if classification_layer_type == "categorical":
        classification_layer = ClassificationLayer(
            hidden_size=hidden_size, num_labels=3, **classification_layer_kwargs
        )
    elif classification_layer_type == "ordered":
        classification_layer = OrderedClassificationLayer(
            hidden_size=hidden_size, num_labels=3, **classification_layer_kwargs
        )

    loss_layer_type = loss_layer["type"]
    loss_layer_kwargs = loss_layer["kwargs"]
    if loss_layer_type == "CE":
        loss_layer = CrossEntropyLossLayer(**loss_layer_kwargs)
    elif loss_layer_type == "BCElogits":
        loss_layer = BCELossWithLogitsLayer(**loss_layer_kwargs)
    return {
        "core_model": core_model,
        "tokenizer": tokenizer,
        "fusion_layer": fusion_layer,
        "representation_layer": representation_layer,
        "classification_layer": classification_layer,
        "loss_layer": loss_layer,
    }


DEFAULT_CONFIG = {
    "fusion_layer": {"type": "select", "kwargs": {}},
    "representation_layer": {"type": "mlm_model", "kwargs": {}},
    "classification_layer": {
        "type": "categorical",
        "kwargs": {"words_init": [], "tied": False},
    },
    "loss_layer": {"type": "CE", "kwargs": {}},
}


def get_model_from_config(model_mlm, tokenizer, config, *args, **kwargs):
    """_summary_

    Args:
        model_mlm (ModelForMaskedLM): model to use
        tokenizer (FastTokenizer): hg tokenizer
        config (dict): configuration file of the model
    """
    core_model, mlm_head = separate_core_head_model(model_mlm)

    fusion_layer = config["fusion_layer"]["type"]

    hidden_size = core_model.config.to_dict()["hidden_size"]

    if fusion_layer == "max_pooling":
        fusion_layer = FilterFusionLayer(MaxPoolingFusionLayer(abs=False))
    elif fusion_layer == "mean":
        fusion_layer = FilterFusionLayer(MeanFusionLayer())
    elif fusion_layer == "max_pooling_abs":
        fusion_layer = FilterFusionLayer(MaxPoolingFusionLayer(abs=True))
    elif fusion_layer == "attention":
        fusion_layer = FilterFusionLayer(
            AttentionFusionLayer(hidden_dim=hidden_size, kq_dim=hidden_size)
        )
    elif fusion_layer == "cls":
        fusion_layer = ClsTokenOnly()
    elif fusion_layer == "select":
        fusion_layer = SelectFusionLayer()
    else:
        raise ValueError(f"Unknown fusion layer {fusion_layer}")

    architecture = core_model.config.to_dict()["architectures"][0]
    if hasattr(core_model, "bert") or architecture.startswith("Bert"):
        arch_name = "Bert"
        arch_mlm = BertMlmRepresentationLayer
    elif (
        hasattr(core_model, "roberta")
        or architecture.startswith("Roberta")
        or architecture.startswith("Camembert")
        or architecture.startswith("XLMRoberta")
    ):
        arch_name = "Roberta"
        arch_mlm = RobertaMlmRepresentationLayer
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    representation_layer = config["representation_layer"]["type"]

    if representation_layer == "mlm_config":
        representation_layer = arch_mlm(from_config=model_mlm.config)
    elif representation_layer == "mlm_model":
        representation_layer = arch_mlm(from_model=model_mlm)
    elif representation_layer == "cls_pooler":
        representation_layer = ClsPooler(config=model_mlm.config)
    elif representation_layer == "none":
        representation_layer = DummyRepresentationLayer()

    classification_layer = config["classification_layer"]["type"]
    if classification_layer == "categorical":
        kwargs_cl = config["classification_layer"]["kwargs"]
        print(kwargs_cl)

        if not "num_labels" in kwargs_cl:
            num_labels = 3
        if "words_init" in kwargs_cl and len(kwargs_cl["words_init"]) > 0:
            num_labels = len(kwargs_cl["words_init"])
            if kwargs_cl["tied"]:
                word_embeddings, indices = get_word_representation_model(
                    model_mlm, tokenizer, words=kwargs_cl["words_init"], tied=True
                )
                print(word_embeddings.shape)
                print(indices)
                init_tied = TiedLinear(word_embeddings, indices)
                classification_layer = ClassificationLayer(
                    hidden_size=hidden_size, num_labels=num_labels, init_tied=init_tied
                )
            else:
                embeddings, indices = get_word_representation_model(
                    model_mlm, tokenizer, words=kwargs_cl["words_init"], tied=False
                )
                if "normalize" in kwargs_cl and kwargs_cl["normalize"]:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                classification_layer = ClassificationLayer(
                    hidden_size=hidden_size,
                    num_labels=num_labels,
                    init_projection_embeddings=embeddings,
                )
        else:
            classification_layer = ClassificationLayer(
                hidden_size=hidden_size, num_labels=3
            )
    elif classification_layer == "ordered":
        kwargs_cl = config["classification_layer"]["kwargs"]
        if (
            "good_words" in kwargs_cl
            and len(kwargs_cl["good_words"]) >= 1
            and len(kwargs_cl["bad_words"]) >= 1
        ):
            init_projection = get_ordered_word_representation_init(
                model_mlm,
                tokenizer,
                good_words=kwargs_cl["good_words"],
                bad_words=kwargs_cl["bad_words"],
            )
            classification_layer = OrderedClassificationLayer(
                hidden_size=hidden_size,
                num_labels=3,
                init_projection_embedding=torch.nn.functional.normalize(
                    init_projection, p=2, dim=1
                ),
            )
        else:
            init_projection = None
            classification_layer = OrderedClassificationLayer(
                hidden_size=hidden_size, num_labels=3
            )

    loss_layer = config["loss_layer"]["type"]
    if loss_layer == "CE":
        loss_layer = CrossEntropyLossLayer(**config["loss_layer"]["kwargs"])
    elif loss_layer == "BCElogits":
        loss_layer = BCELossWithLogitsLayer(**config["loss_layer"]["kwargs"])
    return {
        "core_model": core_model,
        "tokenizer": tokenizer,
        "fusion_layer": fusion_layer,
        "representation_layer": representation_layer,
        "classification_layer": classification_layer,
        "loss_layer": loss_layer,
    }
