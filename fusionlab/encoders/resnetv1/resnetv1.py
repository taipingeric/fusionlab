import torch
import torch.nn as nn

from fusionlab.utils import autopad

# ResNet50
# Ref:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, activation=True, padding=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, autopad(kernel_size))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, cin, dims, kernel_size=3, stride=None):
        super().__init__()
        dim1, dim2, dim3 = dims
        self.conv1 = ConvBlock(cin, dim1, kernel_size=1)
        self.conv2 = ConvBlock(dim1, dim2, kernel_size=kernel_size,
                               stride=stride if stride else 1)
        self.conv3 = ConvBlock(dim2, dim3, kernel_size=1, activation=False)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if not stride else ConvBlock(cin, dim3,
                                                               kernel_size=1,
                                                               stride=stride,
                                                               activation=False)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += identity
        x = self.act(x)
        return x


class ResNet50V1(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(cin, 64, 7, stride=2),
            nn.MaxPool2d(3, 2, padding=autopad(3))
        )
        self.conv2 = nn.Sequential(
            Bottleneck(64, [64, 64, 256], 3, stride=1),
            Bottleneck(256, [64, 64, 256], 3),
            Bottleneck(256, [64, 64, 256], 3),
        )
        self.conv3 = nn.Sequential(
            Bottleneck(256, [128, 128, 512], 3, stride=2),
            Bottleneck(512, [128, 128, 512], 3),
            Bottleneck(512, [128, 128, 512], 3),
            Bottleneck(512, [128, 128, 512], 3),
        )
        self.conv4 = nn.Sequential(
            Bottleneck(512, [256, 256, 1024], 3, stride=2),
            Bottleneck(1024, [256, 256, 1024], 3),
            Bottleneck(1024, [256, 256, 1024], 3),
            Bottleneck(1024, [256, 256, 1024], 3),
            Bottleneck(1024, [256, 256, 1024], 3),
            Bottleneck(1024, [256, 256, 1024], 3),
        )
        self.conv5 = nn.Sequential(
            Bottleneck(1024, [512, 512, 2048], 3, stride=2),
            Bottleneck(2048, [512, 512, 2048], 3),
            Bottleneck(2048, [512, 512, 2048], 3),
        )

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    cin = 128
    inputs = torch.normal(0, 1, (1, cin, 224, 224))

    output = Bottleneck(cin, [64, 64, 128])(inputs)
    shape = list(output.shape)
    print("Bottleneck", shape)
    assert shape == [1, 128, 224, 224]

    output = Bottleneck(cin, [128, 128, 256], stride=1)(inputs)
    shape = list(output.shape)
    print("Bottleneck first conv for aligh dims", shape)
    assert shape == [1, 256, 224, 224]

    output = Bottleneck(cin, [64, 64, 128], stride=2)(inputs)
    shape = list(output.shape)
    print("Bottleneck downsample", shape)
    assert shape == [1, 128, 112, 112]

    output = ResNet50V1(cin)(inputs)
    shape = list(output.shape)
    print("ResNet50V1", shape)
    assert shape == [1, 2048, 7, 7]




