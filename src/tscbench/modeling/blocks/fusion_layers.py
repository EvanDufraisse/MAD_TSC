#
# Created on Thu Aug 18 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch


class AttentionFusionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, kq_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kq_dim = kq_dim
        self.K = torch.nn.Linear(hidden_dim, kq_dim)
        self.Q = torch.nn.Linear(kq_dim, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X, **kwargs):
        X = self.dropout(X)
        K = self.K(X)
        Q = self.Q(K)
        Q = self.softmax(Q)
        O = Q * X
        return torch.sum(O, dim=1)


class FilterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, x_select, **kwargs):
        X_filtered = x_select.unsqueeze(-1) * X
        return X_filtered


class MaxPoolingFusionLayer(torch.nn.Module):
    def __init__(self, abs=True):
        super().__init__()
        self.abs = abs

    def forward(self, X, **kwargs):
        if not (self.abs):
            return torch.max(X, dim=1).values
        else:
            return torch.gather(
                X, 1, torch.argmax(torch.abs(X), dim=1, keepdim=True)
            ).squeeze()


class MeanFusionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, counts, **kwargs):
        sum_X_filtered = torch.sum(X, dim=1)
        mean = sum_X_filtered / counts.unsqueeze(-1)
        return mean


class FilterFusionLayer(torch.nn.Module):
    def __init__(self, fusion_layer):
        super().__init__()
        self.fusion_layer = fusion_layer
        self.filter_layer = FilterLayer()

    def forward(self, X, x_select, counts, **kwargs):
        X_filtered = self.filter_layer(X, x_select)
        out = self.fusion_layer(X=X_filtered, counts=counts)
        return out


class ClsTokenOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, **kwargs):
        return X[:, 0]


class SelectFusionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, classifying_locations, **kwargs):
        return X[classifying_locations[0], classifying_locations[1]]
