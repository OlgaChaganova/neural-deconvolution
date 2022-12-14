from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import Module


class ConvBlock(Module):
    def __init__(self, is_rgb: bool = True, ngpu: int = 1):
        super(ConvBlock, self).__init__()
        self.ngpu = ngpu
        in_channels = 3 if is_rgb else 1
        n_feature_2 = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.Conv2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.Conv2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, in_channels, 5, 1, 2, bias=False),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
