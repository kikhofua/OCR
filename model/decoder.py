import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# from model.init_module import InitModule

# init state model - produces the initial state C_0 of the LSTM Stack, based on the pooled feature vectors
#                    is modeled as a multilayer perceptron with common hidden layers and 2Q distinct output layers
#                    is to setup the decoder to extract relevant features from the image (if any)
#                    before it starts generating the sequence

# attention model

# LSTM Stack

# Deep Output Layer


class AttentionModel(nn.Module):
    def __init__(self, hidden_size, output_depth, output_width, output_height):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = 1000  # length of longest latex snippet
        self.layer1_default_size = 256
        self.layer2_default_size = 128
        self.output_depth = output_depth
        self.output_width = output_width
        self.output_height = output_height
        self.flattened_size = self.output_depth * self.output_width * self.output_height
        self.linear1 = nn.Linear(self.hidden_size + self.flattened_size, max(self.layer1_default_size, self.output_width * self.output_height))
        self.linear2 = nn.Linear(max(self.layer1_default_size, self.output_width * self.output_height), max(self.layer2_default_size, self.output_width * self.output_height))

    def forward(self, encoder_outputs, hidden):
        flattened_encoder_outputs = encoder_outputs.view(-1)
        x = nn.Tanh()
        cat = torch.cat((flattened_encoder_outputs, hidden), 0)
        l1 = self.linear1(cat)
        l1 = x(l1)
        l2 = self.linear2(l1)
        l2 = x(l2)
        attn_weights = F.softmax(l2)
        output = attn_weights * encoder_outputs.view(1, 512, -1)
        return output


class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMStack, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, attention_model_output, hidden, num_layers):
        out = self.rnn(attention_model_output, hidden)
        return out

