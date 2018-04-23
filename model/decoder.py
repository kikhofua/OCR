

# init state model - produces the initial state C_0 of the LSTM Stack, based on the pooled feature vectors
#                    is modeled as a multilayer perceptron with common hidden layers and 2Q distinct output layers
#                    is to setup the decoder to extract relevant features from the image (if any)
#                    before it starts generating the sequence

# attention model

# LSTM Stack

# Deep Output Layer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class DecoderRNN(nn.Module):
     def __init__(self, input_size, hidden_size, pooled_feature_vectors, embedding):
         super(DecoderRNN, self).__init__()
         self.input_size = input_size
         self.hidden_size = hidden_size
         self.pooled_feature_vectors = pooled_feature_vectors
         self.embedding = embedding

     def forward(self, *input):



