from collections import OrderedDict

import numpy as np
import torch.nn as nn

from model.utils import MLPSpecs


class InitModule(nn.Module):
    def __init__(self, mlp_specs: MLPSpecs):
        super(InitModule, self).__init__()
        self.specs = mlp_specs
        self.sequence_layers = self.build_hidden_layers()

    def build_hidden_layers(self) -> nn.Sequential:
        hidden_sizes, output_size, hidden_act_fn, output_act_fn = self.specs
        sequential_layers = []
        for i in range(len(hidden_sizes) - 1):
            name = "hidden_{}".format(i)
            layer = hidden_act_fn(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            sequential_layers.append((name, layer))
        output_layer = output_act_fn(nn.Linear(hidden_sizes[-1], output_size))
        sequential_layers.append(output_layer)
        return nn.Sequential(OrderedDict(sequential_layers))

    def forward(self, H, z, E_y):
        input = np.concatenate((H, z, E_y), axis=0)
        return self.sequence_layers(input)
