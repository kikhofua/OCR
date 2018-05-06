import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMStack, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x, h, c):
        out = self.rnn(x, h, c)
        return out


# bs = 10
# seq_len = 7
# input_size = 28
# hidden_size = 50
# num_layers = 2
#
# test_lstm = LSTMStack(input_size, hidden_size, num_layers)
# print(test_lstm)
#
# input = Variable(torch.randn(seq_len, bs, input_size))
# h0 = Variable(torch.randn(num_layers, bs, hidden_size))
# c0 = Variable(torch.randn(num_layers, bs, hidden_size))
# output, h = test_lstm(input, h0, c0)
# print('output', output.size())
# print('h and c', h[0].size(), h[1].size())