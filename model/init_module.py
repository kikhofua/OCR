from collections import OrderedDict, namedtuple

import torch.nn as nn


VFMDims = namedtuple('VisualFeatureMapDimensions', 'D_tick H W')
MLPSpecs = namedtuple('MultilayeredPerceptronSpecs', 'hidden_sizes output_size hidden_act_fn output_act_fn')


class InitModule(nn.Module):
    def __init__(self, mlp_specs: MLPSpecs, rnn_stack_size: int):
        super(InitModule, self).__init__()
        self.specs = mlp_specs
        self.Q = rnn_stack_size
        self.shared_weights = self.build_shared_weights()
        self.c_0s, self.h_0s = self.build_output_layers()

    def build_output_layers(self):
        hidden_states = []
        cell_states = []
        hidden_sizes, output_size, _, output_act_fn = self.specs
        for _ in range(self.Q):
            cell_state = output_act_fn(nn.Linear(hidden_sizes[-1], output_size))
            cell_states.append(cell_state)
            hidden_state = output_act_fn(nn.Linear(hidden_sizes[-1], output_size))
            hidden_states.append(hidden_state)
        return cell_states, hidden_states

    def build_shared_weights(self) -> nn.Sequential:
        hidden_sizes, _, hidden_act_fn, _ = self.specs
        sequential_layers = []
        for i in range(len(hidden_sizes) - 1):
            name = "hidden_{}".format(i)
            layer = hidden_act_fn(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            sequential_layers.append((name, layer))
        return nn.Sequential(OrderedDict(sequential_layers))

    def forward(self, flattened_a):
        common_out = self.shared_weights(flattened_a)
        c_outputs = []
        h_outputs = []
        for h, c in zip(self.h_0s, self.c_0s):
            c_out = c(common_out)
            c_outputs.append(c_out)
            h_out = h(common_out)
            h_outputs.append(h_out)
        return c_outputs, h_outputs
