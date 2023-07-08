#
# Created on Tue Aug 23 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#

from tscbench.modeling.models.absa.absa_model import AbsaModel
from tscbench.data.load.absa import AbsaModelProcessor
from tokenizers import AddedToken


def add_tokens_to_tokenizer(tokenizer, num_tokens, template="[SOFT_k]"):
    tokens_to_add = [
        AddedToken(template.replace("k", str(i)), lstrip=True)
        for i in range(num_tokens)
    ]
    if hasattr(tokenizer, "do_lower_case"):
        if tokenizer.do_lower_case:
            tokens_to_add = [
                AddedToken(template.replace("k", str(i)).lower(), lstrip=True)
                for i in range(num_tokens)
            ]

    tokenizer.add_tokens(tokens_to_add)
    prompt_template = ""
    return tokenizer, prompt_template


# def add_tokens_to_tokenizer(tokenizer, num_tokens, template=" [SOFT_k]"):
#     tokens_to_add = [
#         AddedToken(template.replace("k", str(i)), lstrip=True)
#         for i in range(num_tokens)
#     ]
#     try:
#         if tokenizer.do_lower_case:
#             tokens_to_add = [token.lower() for token in tokens_to_add]
#     except:
#         pass
#     tokenizer.add_tokens(tokens_to_add)
#     prompt_template = (
#         " ".join([str(tok) for tok in tokens_to_add[0 : len(tokens_to_add) // 2]])
#         + " <entity> "
#         + " ".join([str(tok) for tok in tokens_to_add[len(tokens_to_add) // 2 :]])
#         + " <mask>".replace("  ", " ")
#     )
#     return tokenizer, prompt_template


class SoftCls(AbsaModel):
    def __init__(
        self,
        core_model,
        tokenizer,
        fusion_layer,
        representation_layer,
        classification_layer,
        n_prompt_tokens: int = 2,
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
        self.tokenizer, self.prompt_template = add_tokens_to_tokenizer(
            tokenizer, n_prompt_tokens
        )
        self.processor = AbsaModelProcessor(
            tokenizer=self.tokenizer,
            prompt_template=self.prompt_template,
            replace_by_main_mention=replace_by_main_mention,
            replace_by_special_token=replace_by_special_token,
            sentiment_mapping=sentiment_mapping,
            soft_cls=True,
        )

        self.core_model = core_model
        self.core_model.resize_token_embeddings(len(tokenizer))

        self.fusion_layer = fusion_layer
        self.representation_layer = representation_layer
        self.classification_layer = classification_layer

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
