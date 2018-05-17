import torch
import torch.utils.data
from torch.autograd import Variable


delimiter = None
SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()


class Vocab:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: "SOS", 1: "EOS", 2 : " "}
        self.num_of_tokens = 3  # so far just SOS and EOS

    def add_token(self, snippet):
        for token in snippet.split(delimiter):
            if token not in self.token2index:
                self.token2index[token] = self.num_of_tokens
                self.token2count[token] = 1
                self.index2token[self.num_of_tokens] = token
                self.num_of_tokens += 1
            else:
                self.token2count[token] += 1


# ----------------------------- NOW PREPARE FOR TRAINING -------------------------------- #

def vector_from_tokens(snippet):
    return [Vocab.token2index[token] for token in snippet.split(' ')]


def variable_from_snippet(lang, snippet):
    indexes = vector_from_tokens(snippet)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result