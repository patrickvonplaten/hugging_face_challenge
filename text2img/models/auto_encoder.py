import torch
from torch import nn
from torch.nn import functional as F


class TwoLayerPerceptron(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, dropout_probability):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.dropout_probability = dropout_probability

        self.hidden_layer = nn.Linear(in_features, hidden_dim)
        self.hidden_activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_probability)
        self.output_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, X):
        X = self.hidden_layer(X)
        X = self.hidden_activation(X)
        X = self.dropout(X)
        return self.output_layer(X)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_isntance, decoder_instance):
        super().__init__()
        self.encoder = encoder_isntance
        self.decoder = decoder_instance

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return encoded, decoded
