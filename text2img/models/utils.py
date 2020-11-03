import torch
from torch import nn


class BertSimplePooler(nn.Module):
    """Pools the BERT hidden states by taking the first element of the sequence"""
    def forward(self, X):
        # Select the hidden state corresponding to the [CLS] token
        X = torch.index_select(X, -2, torch.LongTensor([0]))
        return X


class SqueezeLayer(nn.Module):
    """Squeezes input (removes dimensions equal to 1)"""
    def forward(self, X):
        return X.squeeze()
