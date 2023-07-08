#
# Created on Sun Aug 28 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch
from typing import Any


def generate_configurations_bcelosswithlogits(num_labels):
    configurations = []
    for k in range(num_labels):
        configurations.append([1] * (k + 1) + [0] * (num_labels - (k + 1)))
    return configurations


class CrossEntropyLossLayer(object):
    def __init__(self, weighted=False, num_labels=3, *args, **kwargs):
        self.weighted = weighted
        self.num_labels = num_labels

    def setup(self, dataset=None, weights=None):
        if self.weighted:
            assert not (
                (dataset is None) and (weights is None)
            ), "dataset or weights must be supplied to perform class balancing"
            if not (weights is None):
                weights = torch.FloatTensor(weights)
                coeff_balance = weights / weights.sum()
                if torch.cuda.is_available():
                    coeff_balance = coeff_balance.cuda()
            else:
                labels = []
                dataset_iter = iter(dataset)
                for batch in dataset_iter:
                    labels.append(batch[3])
                labels = torch.LongTensor(labels)
                unique_counts = torch.unique(labels, return_counts=True)[1]
                coeff_balance = torch.ones(self.num_labels) / unique_counts
                coeff_balance /= coeff_balance.sum()
                coeff_balance = torch.FloatTensor(coeff_balance)
                if torch.cuda.is_available():
                    coeff_balance = coeff_balance.cuda()
            self.loss_layer = torch.nn.CrossEntropyLoss(weight=coeff_balance)
        else:
            self.loss_layer = torch.nn.CrossEntropyLoss()

    def __call__(self, predictions, true_labels, *args: Any, **kwds: Any) -> Any:
        self.forward(predictions, true_labels)

    def forward(self, predictions, true_labels):
        return self.loss_layer(predictions, true_labels)


class BCELossWithLogitsLayer(object):
    def __init__(self, num_labels=3, *args, **kwargs):
        self.loss_layer = torch.nn.BCEWithLogitsLoss()
        self.embs = torch.nn.Embedding(num_labels, num_labels)
        configurations = generate_configurations_bcelosswithlogits(num_labels)
        self.embs.weight = torch.nn.parameter.Parameter(
            torch.FloatTensor(configurations), requires_grad=False
        )

    def setup(self, *args, **kwargs):
        pass

    def __call__(self, predictions, true_labels, *args: Any, **kwds: Any) -> Any:
        self.forward(predictions, true_labels)

    def forward(self, predictions, true_labels):
        true_labels = self.embs(true_labels)
        return self.loss_layer(predictions, true_labels)
