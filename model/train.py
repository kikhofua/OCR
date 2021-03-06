# from model.evaluate import *
import os
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
from torch.utils.data import DataLoader
import argparse
from model.encoder import *
from model.decoder import *
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from torchvision.transforms import ToTensor
from data_processing.data_loader import *
from model.deep_output import *


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


def train(args, vocab):
    latex_img_data = ImageSnippetDataset(args.latex_dir, args.image_dir)
    torch_data_loader = DataLoader(latex_img_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Build the models
    embeddings = nn.Embedding(541, 300)
    encoder = EncoderCNN(args.ngpu)
    attention_model = AttentionModel(args.hidden_size, args.output_depth, args.output_width, args.output_height)
    attention_hidden = Variable(torch.zeros(1500), volatile=True)
    lstm_stack = LSTMStack(args.input_size, args.hidden_size, args.num_layers)

    # attn_layer_size, encoder_output_size, embedding_size
    attn_layer_size = 2
    encoder_output_size = 2
    embedding_size = (541, 300)
    deep_output_layer = DeepOutputLayer(attn_layer_size, encoder_output_size, embedding_size)

    if torch.cuda.is_available():
        encoder.cuda()
        attention_model.cuda()
        lstm_stack.cuda()

    # Loss and Optimizer
    loss = 0
    criterion = nn.CrossEntropyLoss()
    params = list(attention_model.parameters()) + list(encoder.parameters()) + list(lstm_stack.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(torch_data_loader)
    print("total step", total_step)
    current_token = "<SOS>"
    previous_index = vocab[current_token]

    for epoch in range(args.num_epochs):
        for i, (img_tensors, targets) in enumerate(torch_data_loader):
            print("i", i)
            print(targets)
            features = encoder(img_tensors)
            attention_outputs = attention_model(features, attention_hidden)
            print(type(attention_outputs))
            lstm_output, hidden_tuples = lstm_stack(attention_outputs, lstm_hidden, embeddings(torch.LongTensor([previous_index])))

            # output = None
            # attn_output_size, encoder_output_size, embedding_size

            output = deep_output_layer(attention_outputs, embeddings[previous_index], lstm_output)
            topv, topi = output.data.topk(1)
            ni = topi[0][0]

            previous_index = ni

            loss = criterion(output, Variable(targets, volatile=True))
            attention_model.zero_grad()
            encoder.zero_grad()
            lstm_stack.zero_grad()
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

                torch.save(attention_model.state_dict(),
                           os.path.join(args.model_path,
                                        'attention_model-%d-%d.pkl' % (epoch + 1, i + 1)))

                torch.save(lstm_stack.state_dict(),
                           os.path.join(args.model_path,
                                        'lstm_stack-%d-%d.pkl' % (epoch + 1, i + 1)))


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
    parser.add_argument('--input_size', type=int, default=858,
                        help='dimension of word embedding vectors')

    parser.add_argument('--output_depth', type=int, default=512,
                        help='dimension of output size')

    parser.add_argument('--output_width', type=int, default=33,
                        help='dimension of output size')

    parser.add_argument('--output_height', type=int, default=26,
                        help='dimension of output size')

    parser.add_argument('--hidden_size', type=int, default=1500,
                        help='dimension of lstm hidden states')
    # HIDDEN SIZE ????

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--ngpu', type=int, default=2)

    latex_path = "/Users/Kamoya/OCR/data/latex_snippets/"
    imgs_path = "/Users/Kamoya/OCR/data/img_snippets/"

    args = parser.parse_args()
    data = ImageSnippetDataset(args.latex_dir, args.image_dir)
    vocab = data.get_vocab()
    # print(vocab["<SOS>"])
    train(args, vocab)












