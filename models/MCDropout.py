import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    def __init__(self, n_layers, p):
        super(MCDropout, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(n_layers[:-1], n_layers[1:])
        ])
        self.p = p

    def forward(self, x):
        # do not perform dropout on the input layer
        x = F.relu(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = F.relu(layer(F.dropout(x, self.p, training=True)))

        x = self.layers[-1](F.dropout(x, self.p, training=True))

        return x
