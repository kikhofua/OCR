# from model.evaluate import *
import os
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import argparse
from data_processing import data_loader as dl
from model.encoder import *
from model.decoder import *
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from torchvision.transforms import ToTensor
from data_processing.data_loader import *

attention_hidden = None
lstm_hidden = None

# opt = parser.parse_args()
# print(opt)

# latex_dataset_dir = "/Users/Kamoya/OCR/data/latex_snippets"
device = torch.cuda.is_available()
# ngpu = int(args.ngpu)
# nz = int(opt.nz)
# ngf = int(opt.ngf)
# ndf = int(opt.ndf)
# nc = 3

# if opt.image_dir in ['img_snippets']:
#     # folder dataset
#     img_dataset = dset.ImageFolder(root=opt.img_dir)
# assert img_dataset


def train(args):
    latex_img_data = ImageSnippetDataset(args.latex_dir, args.image_dir)
    data_loader = DataLoader(latex_img_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.ngpu)
    decoder1 = AttentionModel(args.hidden_size, args.output_size)
    decoder2 = LSTMStack(args.input_size, args.hidden_size, args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder1.cuda()
        decoder2.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder1.parameters()) + list(encoder.parameters()) + list(decoder2.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    print("total step", total_step)
    for epoch in range(args.num_epochs):
        for i, (img_tensors, targets) in enumerate(data_loader):
            print("i", i)
            # targets = pack_padded_sequence(targets, batch_first=True)[0]
            decoder1.zero_grad()
            encoder.zero_grad()
            features = encoder(img_tensors)
            attention_outputs = decoder1(features, attention_hidden)
            lstm_outputs = decoder2(attention_outputs, lstm_hidden, num_layers=2)

            loss = criterion(lstm_outputs, targets)
            loss.backward()
            optimizer.step()

            # # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

            # # Print step info
            if (i+1) % args.save_step == 0:
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))

                torch.save(decoder1.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder1-%d-%d.pkl' % (epoch + 1, i + 1)))

                torch.save(decoder2.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder2-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')

    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')

    parser.add_argument('--image_dir', type=str, default='/Users/Kamoya/OCR/data/img_snippets/',
                        help='directory for processed images')

    parser.add_argument('--latex_dir', type=str,
                        default='/Users/Kamoya/OCR/data/latex_snippets/',
                        help='path for latex snippets')

    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')

    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters

    # INPUT SIZE ????
    parser.add_argument('--input_size', type=int, default=256,
                        help='dimension of word embedding vectors')

    parser.add_argument('--output_size', type=int, default=256,
                        help='dimension of output size')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    # HIDDEN SIZE ????

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--ngpu', type=int, default=2)

    args = parser.parse_args()
    train(args)












