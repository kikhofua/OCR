import torch
import torch.nn as nn
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, ngpu):
        super(EncoderCNN, self).__init__()
        self.H = 315  # input image height
        self.W = 835  # input image weight

        self.num_convolutions = 5  # number of convolutions, maxpool sequences
        self.average_pool_kernel = (1, 1)
        self.average_pool_stride = self.average_pool_kernel
        self.channels = 64

        self.H_prime = 0
        self.W_prime = 0
        self.D_prime = 0

        self.ngpu = ngpu

        """
        Use the nn package to define our model as a sequence of layers. nn.Sequential
        is a Module which contains other Modules, and applies them in sequence to
        produce its output.
        """

        # self.main = nn.Sequential(
        #     """
        #     input is h x w
        #     encode the image into a rectangular visual feature map, A', of a H' x W' visual feature vectors a_(h',w')
        #     of size D'.
        #     all convolution kernels have shape (3,3)
        #     tanh non-linearity
        #     maxpooling shape (2,2) and stride (2,2)
        #     """
        #     nn.Conv2d(1, self.channels, (3, 3), (1, 1)),
        #     nn.Tanh(),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #
        #     nn.Conv2d(self.channels, self.channels*2, (3, 3), (1, 1)),
        #     nn.Tanh(),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #
        #     nn.Conv2d(self.channels * 2, self.channels * 4, (3, 3), (1, 1)),
        #     nn.Tanh(),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #
        #     nn.Conv2d(self.channels * 4, self.channels * 8, (3, 3), (1, 1)),
        #     nn.Tanh(),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #
        #     nn.Conv2d(self.channels * 8, self.channels * 8, (3, 3), (1, 1)),
        #     nn.Tanh(),
        #     nn.MaxPool2d((2, 2), (2, 2)),
        #
        #     """ afterwards, the visual feature vectors are pooled together in rectangular blocks of shape [S_h, S_w]
        #          resulting in pooled feature vectors a_(h,w) of size D = D x S_h x S_w
        #          S_h and S_w are strides by which the pooling operation moves along the height and width of A.
        #          output dimensions after pooling will be: [H,W], H = H'/S_h, W = W'/S_w"""
        #
        #     nn.AvgPool3d((self.channels * 8, 1, 1))
        # )

        self.conv1 = nn.Conv2d(1, self.channels, (3, 3), (1, 1))
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(self.channels, self.channels * 2, (3, 3), (1, 1))
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = nn.Conv2d(self.channels * 2, self.channels * 4, (3, 3), (1, 1))
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv4 = nn.Conv2d(self.channels * 4, self.channels * 8, (3, 3), (1, 1))
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv5 = nn.Conv2d(self.channels * 8, self.channels * 8, (3, 3), (1, 1))
        self.act5 = nn.Tanh()
        self.pool5 = nn.MaxPool2d((2, 2), (2, 2))

        self.feature_pool = nn.AvgPool3d((1, 1, 1))

    def forward(self, input):
        input = Variable(input)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.gpu))
        else:
            # output = self.main(input)
            h = self.pool1(self.act1(self.conv1(input)))
            h = self.pool2(self.act2(self.conv2(h)))
            h = self.pool3(self.act3(self.conv3(h)))
            h = self.pool4(self.act4(self.conv4(h)))
            h = self.pool5(self.act5(self.conv5(h)))
            output = self.feature_pool(h)
        return output




