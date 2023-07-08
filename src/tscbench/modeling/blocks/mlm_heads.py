#
# Created on Tue Aug 09 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
from importlib.metadata import requires
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch.nn.functional as F
from transformers.activations import gelu
import math


class TiedLinear(torch.nn.Module):
    def __init__(self, tied_to: torch.nn.Linear, indices: torch.Tensor):
        super().__init__()
        self.weight = tied_to.weight
        self.bias = tied_to.bias
        self.indices = indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight[self.indices], self.bias[self.indices])


class BertMlmHead(torch.nn.Module):
    def __init__(
        self,
        from_config=None,
        from_model=None,
        all_vocabulary=True,
        limited_vocabulary=[],
        custom_embeddings=[],
    ):
        """This class implements a typical Bert classification head, but can allow for more customization.

        Args:
            from_config (dict, optional): Allows to load from a model config a newly inialized mlm head. Defaults to None.
            from_model (model, optional): Allows to load the weights of a classfication head directly from a MLM headed model. Defaults to None.
            all_vocabulary (bool, optional): Classical MLM head with all vocabulary of the model as output projection layer. Defaults to True.
            limited_vocabulary (list, optional): List of token_ids to keep in the vocabulary in the output layer. Defaults to [].
            custom_embeddings (list, optional): List of custom tensors of size [hidden_size] to serve as output projection. Defaults to [].

        """
        super().__init__()
        if (from_config is not None) + (from_model is not None) != 1:
            raise ValueError(
                "You must provide one and only one of the following: from_config, from_model"
            )

        if from_config is not None:
            bertOnlyMlmHead = BertOnlyMLMHead(from_config)
        elif from_model is not None:
            bertOnlyMlmHead = from_model.cls
        if (
            all_vocabulary + len(limited_vocabulary)
            > 0 + len(custom_embeddings)
            > 0
            != 1
        ):
            raise ValueError(
                "You must provide one and only one of the following: all_vocabulary, limited_vocabulary, custom_embeddings"
            )
        bertLmPredictionHead = bertOnlyMlmHead.predictions
        bertPredictionHeadTransform = bertLmPredictionHead.transform
        self.decoder = bertLmPredictionHead.decoder
        self.bias = bertLmPredictionHead.bias
        if self.decoder.bias is None:
            self.decoder.bias = self.bias
        self.dense = bertPredictionHeadTransform.dense
        self.transform_act_fn = bertPredictionHeadTransform.transform_act_fn
        self.LayerNorm = bertPredictionHeadTransform.LayerNorm

        if all_vocabulary and from_config:
            raise ValueError(
                "You cannot use all_vocabulary and from_config at the same time"
            )
        if limited_vocabulary:
            new_decoder = TiedLinear(self.decoder, limited_vocabulary)
            self.decoder = new_decoder
        if custom_embeddings:
            new_decoder = torch.nn.Linear(
                self.dense.in_features, len(custom_embeddings), bias=True
            )
            embs = torch.stack(custom_embeddings, dim=0)
            if embs.shape != new_decoder.weight.shape:
                embs = embs.T
            if embs.shape != new_decoder.weight.shape:
                raise ValueError(
                    "The shape of the custom embeddings is not the same as the shape of the decoder"
                )
            new_decoder.weight = torch.nn.Parameter(embs, requires_grad=True)
            self.decoder = new_decoder
            self.bias = self.decoder.bias

    def forward(self, sequence_output):
        """Ouput the logits of the model.

        Args:
            sequence_output (tensor): output batch of the model.

        Returns:
            tensor: scores over the vocabulary.
        """
        h = self.dense(sequence_output)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h

    def forward_representations(self, sequence_output):
        """Ouput the representation before projection of the model.
        Can allow for more customization (using multiple tokens for an output)
        """
        h = self.dense(sequence_output)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        return h


class RobertaMlmHead(torch.nn.Module):
    def __init__(
        self,
        from_config=None,
        from_model=None,
        all_vocabulary=True,
        limited_vocabulary=[],
        custom_embeddings=[],
    ):
        """This class implements a typical RoBerta classification head, but can allow for more customization.

        Args:
            from_config (dict, optional): Allows to load from a model config a newly inialized mlm head. Defaults to None.
            from_model (model, optional): Allows to load the weights of a classfication head directly from a MLM headed model. Defaults to None.
            all_vocabulary (bool, optional): Classical MLM head with all vocabulary of the model as output projection layer. Defaults to True.
            limited_vocabulary (list, optional): List of token_ids to keep in the vocabulary in the output layer. Defaults to [].
            custom_embeddings (list, optional): List of custom tensors of size [hidden_size] to serve as output projection. Defaults to [].

        """
        super().__init__()
        if (from_config is not None) + (from_model is not None) != 1:
            raise ValueError(
                "You must provide one and only one of the following: from_config, from_model"
            )

        if from_config is not None:
            config = from_config
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.layer_norm = torch.nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

            self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size)
            self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
            self.decoder.bias = self.bias

        elif from_model is not None:
            self.dense = from_model.lm_head.dense
            self.layer_norm = from_model.lm_head.layer_norm
            self.decoder = from_model.lm_head.decoder
            self.bias = from_model.lm_head.bias
            self.decoder.bias = self.bias

        if all_vocabulary and from_config:
            raise ValueError(
                "You cannot use all_vocabulary and from_config at the same time"
            )
        if limited_vocabulary:
            new_decoder = TiedLinear(self.decoder, limited_vocabulary)
            self.decoder = new_decoder
        if custom_embeddings:
            new_decoder = torch.nn.Linear(
                self.dense.in_features, len(custom_embeddings), bias=True
            )
            embs = torch.stack(custom_embeddings, dim=0)
            if embs.shape != new_decoder.weight.shape:
                embs = embs.T
            if embs.shape != new_decoder.weight.shape:
                raise ValueError(
                    "The shape of the custom embeddings is not the same as the shape of the decoder"
                )
            new_decoder.weight = torch.nn.Parameter(embs, requires_grad=True)
            self.decoder = new_decoder
            self.bias = self.decoder.bias

    def forward(self, sequence_output):
        """Ouput the logits of the model.

        Args:
            sequence_output (tensor): output batch of the model.

        Returns:
            tensor: scores over the vocabulary.
        """
        h = self.dense(sequence_output)
        h = gelu(h)
        h = self.layer_norm(h)
        h = self.decoder(h)
        return h

    def forward_representations(self, sequence_output):
        """Ouput the representation before projection of the model.
        Can allow for more customization (using multiple tokens for an output)
        """
        h = self.dense(sequence_output)
        h = gelu(h)
        h = self.layer_norm(h)
        return h
