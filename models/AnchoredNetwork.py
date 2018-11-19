import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AnchoredNetwork(nn.Module):
    def __init__(self, n_layers, init_weight_std, init_bias_std):
        super(AnchoredNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(n_layers[:-1], n_layers[1:])
        ])

        w_normal = torch.distributions.normal.Normal(0, init_weight_std)
        b_normal = torch.distributions.normal.Normal(0, init_bias_std)

        # list of tuples (weight, bias) that contains the initial weights
        self.init_weights = []

        for layer in self.layers:
            # sample initial weights
            weight_init = w_normal.sample(layer.weight.size())
            bias_init = b_normal.sample(layer.bias.size())

            # initialize weights
            layer.weight.data.copy_(weight_init)
            layer.bias.data.copy_(bias_init)

            # save weights for later use
            self.init_weights.append((weight_init, bias_init))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)

    def get_weight_diff(self):
        sum = 0

        for i in range(len(self.layers)):
            sum += torch.sum((self.layers[i].weight - self.init_weights[i][0])
                             **2)
            sum += torch.sum((self.layers[i].bias - self.init_weights[i][1])
                             **2)

        return sum
