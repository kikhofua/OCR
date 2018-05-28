from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F


class DeepOutputLayer(nn.Module):
    def __init__(self, attn_layer_size, encoder_output_size, embedding_size):
        """ embedding_size: a tuple (541, 300) """
        super(DeepOutputLayer, self).__init__()
        self.attn_layer_size = attn_layer_size
        self.encoder_output_size = encoder_output_size
        self.linear1 = nn.Linear(self.attn_layer_size + self.encoder_output_size + embedding_size[1], embedding_size[1])
        self.linear2 = nn.Linear(self.attn_layer_size + self.encoder_output_size + embedding_size[1], embedding_size[1])

    def forward(self, attn_layer_output, embeddings, lstm_outputs):
        x = nn.Tanh()
        cat1 = torch.cat((attn_layer_output, embeddings), 0)
        cat2 = torch.cat((cat1, lstm_outputs), 0)
        l1 = self.linear1(cat2)
        l1 = x(l1)
        l2 = self.linear2(l1)
        l2 = x(l2)
        output = F.softmax(l2)
        return output



# class TokenEmbedder(nn.Module):
#     def __init__(self, mlp_specs: MLPSpecs):
#         super(TokenEmbedder, self).__init__()
#         self.specs = mlp_specs
#         self.sequence_layers = self.build_hidden_layers()
#
#     def build_hidden_layers(self) -> nn.Sequential:
#         hidden_sizes, output_size, hidden_act_fn, output_act_fn = self.specs
#         sequential_layers = []
#         for i in range(len(hidden_sizes) - 1):
#             name = "hidden_{}".format(i)
#             layer = hidden_act_fn(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
#             sequential_layers.append((name, layer))
#         output_layer = output_act_fn(nn.Linear(hidden_sizes[-1], output_size))
#         sequential_layers.append(output_layer)
#         return nn.Sequential(OrderedDict(sequential_layers))
#
#     def forward(self, token):
#         return self.sequence_layers(token)
