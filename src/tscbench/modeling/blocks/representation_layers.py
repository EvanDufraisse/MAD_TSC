#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
"""
Representation Layers are intermediate layers used to compute a new representation from the tokens output

"""

import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.activations import gelu


class DummyRepresentationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class BertMlmRepresentationLayer(torch.nn.Module):
    def __init__(self, from_config=None, from_model=None):
        """This class implements the projection layer of the Bert MLM head.

        Args:
            from_config (dict, optional): Allows to load from a model config a newly inialized mlm head. Defaults to None.
            from_model (model, optional): Allows to load the weights of a classfication head directly from a MLM headed model. Defaults to None.
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
        bertLmPredictionHead = bertOnlyMlmHead.predictions
        bertPredictionHeadTransform = bertLmPredictionHead.transform
        self.dense = bertPredictionHeadTransform.dense
        self.transform_act_fn = bertPredictionHeadTransform.transform_act_fn
        self.LayerNorm = bertPredictionHeadTransform.LayerNorm

    def forward(self, sequence_output):
        """Ouput the representation before projection of the model.
        Can allow for more customization (using multiple tokens for an output)
        """
        h = self.dense(sequence_output)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        return h


class RobertaMlmRepresentationLayer(torch.nn.Module):
    def __init__(self, from_config=None, from_model=None):
        """This class implements the projection layer of the Roberta MLM head.

        Args:
            from_config (dict, optional): Allows to load from a model config a newly inialized mlm head. Defaults to None.
            from_model (model, optional): Allows to load the weights of a classfication head directly from a MLM headed model. Defaults to None.
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

        elif from_model is not None:
            self.dense = from_model.lm_head.dense
            self.layer_norm = from_model.lm_head.layer_norm

    def forward(self, sequence_output):
        """Ouput the representation before projection of the model.
        Can allow for more customization (using multiple tokens for an output)
        """
        h = self.dense(sequence_output)
        h = gelu(h)
        h = self.layer_norm(h)
        return h


class ClsPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, X):
        return self.activation(self.dense(X))
