#
# Created on Tue Aug 09 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch
from transformers.activations import gelu
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod


class ClassificationLayer(torch.nn.Module, ABC):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

    @abstractmethod
    def forward(self, embeddings):
        pass


class OrderedClassificationHead(ClassificationLayer):
    """
    This classification head is basically a linear layer with a bias term for each label.
    The insight is that in a hierarchical setting the biases are increasingly higher with each level.
    """

    def __init__(self, hidden_size, num_labels, dropout=0.1, double_layer=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        self.double_layer = double_layer
        if self.double_layer:
            self.intermediate_layer = torch.nn.Linear(
                self.hidden_size, self.hidden_size
            )
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-5)
        else:
            self.intermediate_layer = None
            self.layer_norm = None
        self.classifier = torch.nn.Linear(self.hidden_size, 1)
        self.biases = torch.nn.parameter.Parameter(
            data=torch.rand((1, num_labels)), requires_grad=True
        )

    def forward(self, embeddings):
        x = self.dropout(embeddings)
        if self.double_layer:
            x = self.intermediate_layer(x)
            x = gelu(x)
            x = self.layer_norm(x)
        x = self.classifier(x)
        x = x + self.biases
        return torch.sigmoid(x)

    # def forward_loss(self, output, labels):
    #     pred_flatten = output.flatten()
    #     true_flatten = labels.flatten()
    #     return -torch.mean(torch.log(torch.mul(pred_flatten, true_flatten) + torch.mul(1 - pred_flatten, 1-true_flatten)))


class ClassificationHead(ClassificationLayer):
    """
    Plain non hierarchical classification head
    """

    def __init__(self, hidden_size, num_labels, dropout=0.1, double_layer=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        self.double_layer = double_layer
        if self.double_layer:
            self.intermediate_layer = torch.nn.Linear(
                self.hidden_size, self.hidden_size
            )
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-5)
        else:
            self.intermediate_layer = None
            self.layer_norm = None
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, embeddings, labels=None):
        x = self.dropout(embeddings)
        if self.double_layer:
            x = self.intermediate_layer(x)
            x = gelu(x)
            x = self.layer_norm(x)
        x = self.classifier(x)
        return x

    def forward_loss(self, output, labels):
        return self.loss_fct(output, labels)
