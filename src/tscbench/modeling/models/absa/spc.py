#
# Created on Tue Aug 23 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
from tscbench.modeling.models.absa.absa_model import (
    AbsaModel,
    get_model_layers,
)
from tscbench.data.load.absa import AbsaModelProcessor
from tscbench.modeling.blocks.fusion_layers import *
from tscbench.modeling.blocks.representation_layers import *


class SpcModel(AbsaModel):
    def __init__(
        self,
        core_model,
        tokenizer,
        fusion_layer,
        representation_layer,
        classification_layer,
        loss_layer=None,
        replace_by_main_mention=False,
        replace_by_special_token=None,
        sentiment_mapping={2: 0, 4: 1, 6: 2},
    ):
        super().__init__(
            core_model,
            tokenizer,
            fusion_layer,
            representation_layer,
            classification_layer,
            loss_layer,
        )
        self.core_model = core_model
        self.tokenizer = tokenizer
        self.prompt_template = f"{tokenizer.sep_token} <entity>"
        self.fusion_layer = fusion_layer
        self.representation_layer = representation_layer
        self.classification_layer = classification_layer
        self.processor = AbsaModelProcessor(
            prompt_template=self.prompt_template,
            tokenizer=tokenizer,
            replace_by_main_mention=replace_by_main_mention,
            replace_by_special_token=replace_by_special_token,
            sentiment_mapping=sentiment_mapping,
        )

    def data_generator(self):
        return self.processor

    def forward(
        self,
        batch_tokens,
        x_select,
        classifying_locations,
        counts,
        sentiments=None,
        params=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        h = self.core_model(
            batch_tokens, attention_mask=attention_mask
        ).last_hidden_state
        h = self.fusion_layer(
            X=h,
            classifying_locations=classifying_locations,
            x_select=x_select,
            counts=counts,
        )
        h = self.representation_layer(h)
        h = self.classification_layer(h)
        return h


def get_model(model_mlm, tokenizer, configuration="default", **kwargs):
    """
    Available configurations:
    - default: "cls", "cls_pooler", "categorical"
    - ordered: "cls", "cls_pooler", "ordered"

    """
    if configuration == "default":
        parameters = get_model_layers(
            model_mlm=model_mlm,
            tokenizer=tokenizer,
            fusion_layer="cls",
            representation_layer="cls_pooler",
            classification_layer="ordered",
        )
    elif configuration == "ordered":
        parameters = get_model_layers(
            model_mlm=model_mlm,
            tokenizer=tokenizer,
            fusion_layer="cls",
            representation_layer="cls_pooler",
            classification_layer="ordered",
        )
    else:
        raise ValueError(f"Unknown configuration {configuration}")

    return SpcModel(**parameters, **kwargs)
