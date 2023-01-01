import torch
import torch.nn as nn

from fusionlab.utils import autopad

# ResNet50
# Ref:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
# https://github.com/raghakot/keras-resnet/blob/master/README.md

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding=autopad(kernel_size))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if activation is True else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, cin, dims, kernel_size, downsample=False):
        super().__init__()
        dim1, dim2, dim3 = dims
        self.conv1 = ConvBlock(cin, dim1, kernel_size=1)
        self.conv2 = ConvBlock(dim1, dim2, kernel_size=kernel_size,
                               stride=1 if not downsample else 2)
        self.conv3 = ConvBlock(dim2, dim3, kernel_size=1, activation=False)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if not downsample else Downsample(cin, dim3)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += identity
        x = self.act(x)
        return x


class Downsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ConvBlock(cin, cout, kernel_size=1, stride=2, activation=False)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

if __name__ == '__main__':
    inputs = torch.normal(0, 1, (1, 128, 224, 224))
    block = ResidualBlock(128, [64, 64, 128], 3)
    outputs = block(inputs)
    print(outputs.shape)

    inputs = torch.normal(0, 1, (1, 128, 224, 224))
    block = ResidualBlock(128, [64, 64, 128], 3, downsample=True)
    outputs = block(inputs)
    print(outputs.shape)


