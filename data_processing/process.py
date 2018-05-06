


delimiter = None


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