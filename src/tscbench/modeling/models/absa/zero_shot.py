#
# Created on Mon Aug 08 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
"""
Zero-Shot model implementation
"""
import torch
from tscbench.data.load.absa import AbsaModelProcessor
from tscbench.modeling.models.absa.absa_model import (
    AbsaModel,
    get_model_layers,
)


def get_token_ids_representations_word(tokenizer, word, enforce_no_space=False):
    if enforce_no_space:
        return tokenizer.encode(word, add_special_tokens=False)
    else:
        return tokenizer.encode(" " + word.strip(), add_special_tokens=False)


def get_initialisation_bad_neutral_good(
    model_core, tokenizer, words=["bad", "neutral", "good"]
):
    if model_core.config.to_dict()["architectures"][0].startswith("Bert"):
        try:
            word_embeddings = model_core.bert.embeddings.word_embeddings.weight
        except:
            word_embeddings = model_core.embeddings.word_embeddings.weight
    elif model_core.config.to_dict()["architectures"][0].startswith("Roberta"):
        try:
            word_embeddings = model_core.roberta.embeddings.word_embeddings.weight
        except:
            word_embeddings = model_core.embeddings.word_embeddings.weight
    out = []
    for word in words:
        token_id = tokenizer.encode(word, add_special_tokens=False)
        assert len(token_id) == 1
        out.append(word_embeddings[token_id[0]])
    return torch.stack(out)


class ZeroShotModel(AbsaModel):
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
        sentiment_words=["bad", "ok", "good"],
        fusion_prompt_templates="sum_logits",
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

        # projection_embeddings = get_initialisation_bad_neutral_good(self.core_model, self.tokenizer, sentiment_words)
        # self.classification_layer.set_classifier_weights(projection_embeddings)

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
                                        torch.arange(0, self.num_prompt_templates * 3),
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


# class ZeroShotModel(Model):

#     def __init__(
#         self,
#         core_model_name,
#         model_mlm,
#         tokenizer,
#         classifying_embs = None,
#         classification_words = ["good", "ok", "bad"],
#         fusion_type = "mean",
#         prompt_template = "<entity> is <mask>."
#         ):
#         '''
#         Can either set classifying words such as ["good", "ok", "bad"]
#         Or directly set embeddings that would be accounted as parameters of the model

#         '''
#         if not(classifying_embs is None) and not(classification_words is None):
#             raise ValueError("You can only specify one of the two parameters")
#         elif classifying_embs is None and classification_words is None:
#             raise ValueError("You must specify one of the two parameters")

#         super().__init__(core_model_name)
#         self.fusion_type = fusion_type
#         self.prompt_template = prompt_template

#         if 'roberta' in core_model_name:
#             self.model = model_mlm.roberta
#             # word_embeddings_weights = self.model.bert.embeddings.word_embeddings.weight
#         elif 'bert' in core_model_name:
#             self.model = model_mlm.bert
#             # word_embeddings_weights = self.model.roberta.embeddings.word_embeddings.weight
#         else:
#             raise NotImplementedError('{} model is not supported'.format(core_model_name))

#         tokens_ids_per_class = []
#         if classifying_embs is None:
#             for word in classification_words:
#                 tokens_ids_per_class += get_token_ids_representations_word(tokenizer, word, enforce_no_space=True)  # Not implemented if several tokens per class
#         custom_embeddings = [] if classifying_embs is None else classifying_embs

#         if 'roberta' in core_model_name:
#             self.mlm_head = RobertaMlmHead(from_model=model_mlm, limited_vocabulary=tokens_ids_per_class, custom_embeddings=custom_embeddings)
#         elif 'bert' in core_model_name:
#             self.mlm_head = BertMlmHead(from_model=model_mlm, limited_vocabulary=tokens_ids_per_class, custom_embeddings=custom_embeddings)
#         else:
#             raise NotImplementedError('{} model is not supported'.format(core_model_name))

#     def forward(self, input_ids, attention_mask, *args, **kwargs):
#         '''
#         '''
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         output = self.mlm_head(output)
#         return output


def get_model(model_mlm, tokenizer, configuration="default", **kwargs):
    """
    Available configurations:
    - default: "select", "none", "categorical"

    """
    if configuration == "default":
        parameters = get_model_layers(
            model_mlm=model_mlm,
            tokenizer=tokenizer,
            fusion_layer="select",
            representation_layer="mlm_model",
            classification_layer="categorical",
        )
    else:
        raise ValueError(f"Unknown configuration {configuration}")

    return ZeroShotModel(**parameters, **kwargs)


# def get_mlm_head_representation_word(tokenizer, model_word_embeddings, word, enforce_no_space=False, type_fusion="mean"):
#     '''
#     types of fusions:
#     - mean
#     - first
#     - last
#     - mean first + last
#     - max pooling
#     - max abs pooling
#     '''
#     token_ids = get_token_ids_representations_word(tokenizer, word, enforce_no_space)
#     embs = []
#     for token_id in token_ids:
#         embs.append(model_word_embeddings(torch.tensor([token_id])))
#     embs = torch.stack(embs).squeeze()
#     if type_fusion == "mean":
#         return torch.mean(embs, dim=0)
#     elif type_fusion == "first":
#         return embs[0]
#     elif type_fusion == "last":
#         return embs[-1]
#     elif type_fusion == "mean first + last":
#         return torch.mean(embs[[0,-1]], dim=0)
#     elif type_fusion == "max pooling":
#         return torch.max(embs, dim=0).values
#     elif type_fusion == "max abs pooling":
#         return torch.gather(embs, 0, torch.argmax(torch.abs(embs), dim=0, keepdim=True)).squeeze()

# def get_averaging_function(type_fusion="mean"):
#     if type_fusion == "mean":
#         return torch.mean
#     elif type_fusion == "first":
#         return lambda x: x[0]
#     elif type_fusion == "last":
#         return lambda x: x[-1]
#     elif type_fusion == "mean first + last":
#         return lambda x: torch.mean(x[[0,-1]], dim=0)
#     elif type_fusion == "max pooling":
#         return torch.max
#     elif type_fusion == "max abs pooling":
#         return lambda x: torch.gather(x, 0, torch.argmax(torch.abs(x), dim=0, keepdim=True)).squeeze()
