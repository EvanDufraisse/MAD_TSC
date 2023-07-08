#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class TiedLinear(torch.nn.Module):
    def __init__(self, tied_to: torch.nn.Linear, indices: torch.Tensor):
        super().__init__()
        try:
            self.weight = tied_to.weight
        except:
            self.weight = tied_to
        # self.bias = tied_to.bias
        self.indices = indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight[self.indices])


class OrderedClassificationLayer(torch.nn.Module):
    """
    This classification head is basically a linear layer with a bias term for each label.
    The insight is that in a hierarchical setting the biases are increasingly higher with each level.
    """

    def __init__(
        self, hidden_size, num_labels, dropout=0.1, init_projection_embedding=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.hidden_size, 1)
        if not (init_projection_embedding is None):
            self.classifier.weight = torch.nn.Parameter(
                init_projection_embedding.unsqueeze(0)
            )
        self.biases = torch.nn.parameter.Parameter(
            data=torch.rand((1, num_labels)), requires_grad=True
        )

    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.classifier(x)
        x = x + self.biases
        return x


class ClassificationLayer(torch.nn.Module):
    """
    Plain non hierarchical classification head
    """

    def __init__(
        self,
        hidden_size,
        num_labels,
        dropout=0.1,
        init_projection_embeddings=None,
        init_tied: torch.nn.Linear = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        assert (init_projection_embeddings is not None) + (
            init_tied is not None
        ) < 2, "Cannot initialise projection and supply tied weights linear layer at the same time"

        if not (init_tied is None):
            self.classifier = init_tied
        else:
            self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        if not (init_projection_embeddings is None):
            self.classifier.weight = torch.nn.Parameter(init_projection_embeddings)

    def set_classifier_weights(self, projection_embeddings):
        self.classifier.weight = torch.nn.Parameter(projection_embeddings)

    def forward(self, embeddings, labels=None):
        x = self.dropout(embeddings)
        x = self.classifier(x)
        return x
