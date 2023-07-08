#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
from turtle import forward
import torch
from tscbench.modeling.models.absa.absa_model import AbsaModel, get_model_layers
from tscbench.data.load.absa import AbsaModelProcessor


class PromptModel(AbsaModel):
    def __init__(
        self,
        core_model,
        tokenizer,
        fusion_layer,
        representation_layer,
        classification_layer,
        loss_layer=None,
        prompt_template="<entity> is <mask>.",
        replace_by_main_mention=False,
        replace_by_special_token=None,
        sentiment_mapping={2: 0, 4: 1, 6: 2},
        fusion_prompt_templates="sum_logits",  # sum_probabilities
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
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            replace_by_main_mention=replace_by_main_mention,
            replace_by_special_token=replace_by_special_token,
            sentiment_mapping=sentiment_mapping,
        )
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer
        self.representation_layer = representation_layer
        self.classification_layer = classification_layer
        self.fusion_layer = fusion_layer
        self.core_model = core_model
        if type(prompt_template) is list:
            self.num_prompt_templates = len(prompt_template)
        else:
            self.num_prompt_templates = 1
        self.fusion_prompt_templates = fusion_prompt_templates
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
        if self.num_prompt_templates > 1:
            splits = torch.split(h, self.num_prompt_templates, dim=0)
            # return splits

            if self.fusion_prompt_templates == "sum_logits":
                if self.ordered_classif:
                    h = torch.vstack([torch.sum(split, dim=0) for split in splits])
                else:
                    h = torch.vstack(
                        [
                            torch.sum(
                                torch.vstack(
                                    torch.tensor_split(
                                        elem[
                                            torch.div(
                                                torch.arange(
                                                    0, self.num_prompt_templates * 3
                                                ),
                                                3,
                                                rounding_mode="trunc",
                                            ),
                                            torch.arange(
                                                0, self.num_prompt_templates * 3
                                            ),
                                        ],
                                        self.num_prompt_templates,
                                        dim=0,
                                    )
                                ),
                                dim=0,
                            )
                            for elem in splits
                        ]
                    )
            elif self.fusion_prompt_templates == "sum_probabilities":
                if self.ordered_classif:
                    h = torch.vstack(
                        [
                            torch.sum(torch.softmax(split, dim=1), dim=0)
                            for split in splits
                        ]
                    )
                else:
                    h = torch.vstack(
                        [
                            torch.mean(
                                torch.softmax(
                                    torch.vstack(
                                        torch.tensor_split(
                                            elem[
                                                torch.div(
                                                    torch.arange(
                                                        0, self.num_prompt_templates * 3
                                                    ),
                                                    3,
                                                    rounding_mode="trunc",
                                                ),
                                                torch.arange(
                                                    0, self.num_prompt_templates * 3
                                                ),
                                            ],
                                            self.num_prompt_templates,
                                            dim=0,
                                        )
                                    ),
                                    dim=1,
                                ),
                                dim=0,
                            )
                            for elem in splits
                        ]
                    )
            else:
                raise ValueError(
                    "fusion_prompt_templates should be either sum_logits or sum_probabilities"
                )
        return h


def get_model(model_mlm, tokenizer, configuration="default", **kwargs):
    """
    - defaults: "select", "mlm_model", "categorical"
    - ordered: "select", "mlm_model", "ordered"
    """

    if configuration == "default":
        parameters = get_model_layers(
            model_mlm,
            tokenizer,
            fusion_layer="select",
            representation_layer="mlm_model",
            classification_layer="categorical",
        )
    elif configuration == "ordered":
        parameters = get_model_layers(
            model_mlm,
            tokenizer,
            fusion_layer="select",
            representation_layer="mlm_model",
            classification_layer="ordered",
        )
    else:
        raise ValueError(f"Configuration {configuration} not available")
    return PromptModel(**parameters, **kwargs)
