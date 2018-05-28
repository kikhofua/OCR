import torch
from torch.utils.data import Dataset
# from torch.autograd import Variable
# from torchvision import datasets
import os, os.path
from PIL import Image
from torchvision.transforms import ToTensor


class ImageSnippetDataset(Dataset):
    """Reutrns (image, snippet_vector) pairs"""
    def __init__(self, latex_path, imgs_path):
        self.latex_path = latex_path
        self.imgs_path = imgs_path
        self.v = Vocabulary()
        self.length = 0
        self.max_sequence_length = 1000

    def __len__(self):
        return len([name for name in os.listdir(self.latex_path) if name.endswith(".txt")])

    def __getitem__(self, index):
        """Returns one pair of  (image, snippet_vector) pair"""
        # Converts image to tensor

        img_filename = str(index) + "_sn.jpg"
        self.tensor = to_tensor(img_filename, self.imgs_path)
        img_tensor = self.tensor

        latex_filename = str(index) + "_sn.txt"
        latex_file_path = self.latex_path + str(latex_filename)

        f = open(latex_file_path, "r")
        latex_index_vector = [self.v.add_word("<SOS>")]

        # Converts latex_file to a vector of token indices
        for line in f:
            if line == "\\begin{document}":
                pass
            elif len(line) > 0:
                latex_index_vector.extend([self.v.add_word(w) for w in line])
                latex_index_vector.append(self.v.add_word("<EOS>"))

        padding = [0] * (self.max_sequence_length - len(latex_index_vector))
        latex_index_vector.extend(padding)
        self.length = len(latex_index_vector)

        target = torch.LongTensor(latex_index_vector)
        if torch.cuda.is_available():
            target = target.cuda()
        return img_tensor, target

    def get_vocab(self):
        return self.v.get_vocab()


def to_tensor(x, path):
    """Converts an object (image or tensor or list) to a Pytorch Variable"""
    path_to_x = path + str(x)
    img = Image.open(path_to_x)
    img = ToTensor()(img).unsqueeze(0)
    img = img.squeeze(1)
    if torch.cuda.is_available():
        img = img.cuda()
    return img


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<SOS>": 1, "<EOS>": 2}
        self.idx2word = {1: "<SOS>", 2: "<EOS>"}
        self.idx = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            word_idx = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        else:
            word_idx = self.word2idx[word]
        return word_idx

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def get_vocab(self):
        return self.word2idx


if __name__ == '__main__':
    latex_path = "/Users/Kamoya/OCR/data/latex_snippets/"
    imgs_path = "/Users/Kamoya/OCR/data/img_snippets/"

    data = ImageSnippetDataset(latex_path, imgs_path)
    print("Getting size of directory...")
    print("Length", data.__len__())

    print("Getting a particular item...")
    print(data.__getitem__(10))

