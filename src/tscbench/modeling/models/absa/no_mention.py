#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
"""
Implementation of the NoMentionModel, this is a baseline model that doesn't have access to the particular entity mention an entry is referring to.
The intuition is that the training of such a model can allow to distinguish performance related to general sentiment analysis from performance 
related to targeted entity sentiment analysis.

"""
from tscbench.modeling.models.absa.absa_model import AbsaModel
from tscbench.data.load.absa import AbsaModelProcessor


class NoMentionModel(AbsaModel):
    """
    Implementation of NoMentionModel.
    Fusion layer uses the CLS token only.
    """

    def __init__(
        self,
        core_model,
        tokenizer,
        fusion_layer,
        representation_layer,
        classification_layer,
        replace_by_main_mention=False,
        replace_by_special_token=None,
        loss_layer=None,
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
        self.processor = AbsaModelProcessor(
            tokenizer=tokenizer, sentiment_mapping=sentiment_mapping
        )
        self.tokenizer = tokenizer
        self.representation_layer = representation_layer
        self.classification_layer = classification_layer
        self.fusion_layer = fusion_layer
        self.core_model = core_model
        self.ordered_classif = (
            self.classification_layer.__class__.__name__ == "OrderedClassificationLayer"
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
        **kwargs
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
