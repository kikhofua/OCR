import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, opt):
        super(EncoderCNN, self).__init__()
        self.H = 0  # input image height
        self.W = 0  # input image weight

        self.num_convolutions = 5  # number of convolutions, maxpool sequences
        self.average_pool_kernel = (self.H / 2 ^ self.num_convolutions, self.W / 2 ^ self.num_convolutions)
        self.average_pool_stride = self.average_pool_kernel
        self.channels = 64

        self.H_prime = 0
        self.W_prime = 0
        self.D_prime = 0

        self.gpu = None

        self.main = nn.Sequential(
            # input is h x w

            # encode the image into a rectangular visual feature map, A', of a H' x W' visual feature vectors a_(h',w')
            # of size D'.

            # all convolution kernels have shape (3,3)
            # tanh non-linearity
            # maxpooling shape (2,2) and stride (2,2)

            nn.Conv2d(1, self.channels, (3, 3), (1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(self.channels, self.channels*2, (3, 3), (1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(self.channels * 2, self.channels * 4, (3, 3), (1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(self.channels * 4, self.channels * 8, (3, 3), (1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(self.channels * 4, self.channels * 8, (3, 3), (1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (2, 2)),

            # afterwards, the visual feature vectors are pooled together in rectangular blocks of shape [S_h, S_w]
            #     resulting in pooled feature vectors a_(h,w) of size D = D x S_h x S_w
            #     S_h and S_w are strides by which the pooling operation moves along the height and width of A.
            #     output dimensions after pooling will be: [H,W], H = H'/S_h, W = W'/S_w

            nn.AvgPool3d(self.average_pool_kernel, self.average_pool_stride)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.gpu))
        else:
            output = self.main(input)
        return output