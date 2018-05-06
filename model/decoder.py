import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# init state model - produces the initial state C_0 of the LSTM Stack, based on the pooled feature vectors
#                    is modeled as a multilayer perceptron with common hidden layers and 2Q distinct output layers
#                    is to setup the decoder to extract relevant features from the image (if any)
#                    before it starts generating the sequence

# attention model

# LSTM Stack

# Deep Output Layer


class AttentionModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(self.hidden_size + output_size, self.max_length)

    def forward(self, encoder_outputs, hidden):
        x = nn.Tanh()
        cat = torch.cat((encoder_outputs, hidden), 1)
        l1 = self.linear(cat)
        l1 = x(l1)
        l2 = self.linear(l1)
        l2 = x(l2)
        attn_weights = F.softmax(l2)
        output = np.sum(attn_weights * encoder_outputs)
        return output


class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMStack, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x, h, c):
        out = self.rnn(x, h, c)
        return out

