import torch
import torch.nn as nn
from fusionlab.layers.factories import ConvND, MaxPool, BatchNorm

from fusionlab.utils import autopad

# ResNet50
# Ref:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, spatial_dims=2, stride=1, activation=True, padding=True):
        super().__init__()
        self.conv = ConvND(spatial_dims, cin, cout, kernel_size, stride, autopad(kernel_size))
        self.bn = BatchNorm(spatial_dims, cout)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, cin, dims, kernel_size=3, spatial_dims=2, stride=None):
        super().__init__()
        dim1, dim2 = dims
        self.conv1 = ConvBlock(cin, dim1, 1, spatial_dims)
        self.conv2 = ConvBlock(dim1, dim1, kernel_size, spatial_dims,
                               stride=stride if stride else 1)
        self.conv3 = ConvBlock(dim1, dim2, 1, spatial_dims, activation=False)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if not stride else ConvBlock(cin, dim2, 
                                                               1,
                                                               spatial_dims,
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


class Stem(nn.Sequential):
    def __init__(self, cin, spatial_dims=2):
        super().__init__(
            ConvBlock(cin, 64, 7, spatial_dims, stride=2),
            MaxPool(spatial_dims, 3, 2, padding=autopad(3))
        )


class StageBlock(nn.Sequential):
    def __init__(self, cin, dims, stride, repeats):
        super().__init__()


class ResNet50V1(nn.Module):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__()
        self.conv1 = Stem(cin, spatial_dims)
        self.conv2 = nn.Sequential(
            Bottleneck(64, [64, 256], 3, spatial_dims, stride=1),
            Bottleneck(256, [64, 256], 3, spatial_dims),
            Bottleneck(256, [64, 256], 3, spatial_dims),
        )
        self.conv3 = nn.Sequential(
            Bottleneck(256, [128, 512], 3, spatial_dims, stride=2),
            Bottleneck(512, [128, 512], 3, spatial_dims),
            Bottleneck(512, [128, 512], 3, spatial_dims),
            Bottleneck(512, [128, 512], 3, spatial_dims),
        )
        self.conv4 = nn.Sequential(
            Bottleneck(512, [256, 1024], 3, spatial_dims, stride=2),
            Bottleneck(1024, [256, 1024], 3, spatial_dims),
            Bottleneck(1024, [256, 1024], 3, spatial_dims),
            Bottleneck(1024, [256, 1024], 3, spatial_dims),
            Bottleneck(1024, [256, 1024], 3, spatial_dims),
            Bottleneck(1024, [256, 1024], 3, spatial_dims),
        )
        self.conv5 = nn.Sequential(
            Bottleneck(1024, [512, 2048], 3, spatial_dims, stride=2),
            Bottleneck(2048, [512, 2048], 3, spatial_dims),
            Bottleneck(2048, [512, 2048], 3, spatial_dims),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    cin = 128
    inputs = torch.normal(0, 1, (1, cin, 224, 224))

    output = Bottleneck(cin, [64, 128], spatial_dims=2)(inputs)
    shape = list(output.shape)
    print("Bottleneck", shape)
    assert shape == [1, 128, 224, 224]

    output = Bottleneck(cin, [128, 256], spatial_dims=2, stride=1)(inputs)
    shape = list(output.shape)
    print("Bottleneck first conv for aligh dims", shape)
    assert shape == [1, 256, 224, 224]

    output = Bottleneck(cin, [64, 128], spatial_dims=2, stride=2)(inputs)
    shape = list(output.shape)
    print("Bottleneck downsample", shape)
    assert shape == [1, 128, 112, 112]

    output = ResNet50V1(cin, spatial_dims=2)(inputs)
    shape = list(output.shape)
    print("ResNet50V1", shape)
    assert shape == [1, 2048, 7, 7]

    print("1D ResNet50V1")
    inputs = torch.normal(0, 1, (1, cin, 224))
    output = ResNet50V1(cin, spatial_dims=1)(inputs)
    shape = list(output.shape)
    print("ResNet50V1", shape)
    assert shape == [1, 2048, 7]

    print("3D ResNet50V1")
    D = H = W = 64
    inputs = torch.rand(1, 3, D, H, W)
    model = ResNet50V1(3, spatial_dims=3)
    output = model(inputs)
    print(output.shape)


