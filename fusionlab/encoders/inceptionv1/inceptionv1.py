import torch
import torch.nn as nn

from fusionlab.utils import autopad

# ref: https://arxiv.org/abs/1409.4842
# Going Deeper with Convolutions
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding=autopad(kernel_size))
        # self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding=kernel_size//2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, cin, dim0, dim1, dim2, dim3):
        super().__init__()
        self.branch1 = ConvBlock(cin, dim0)
        self.branch3 = nn.Sequential(ConvBlock(cin, dim1[0], 1),
                                     ConvBlock(dim1[0], dim1[1], 3))
        self.branch5 = nn.Sequential(ConvBlock(cin, dim2[0], 1),
                                     ConvBlock(dim2[0], dim2[1], 5))
        self.pool = nn.Sequential(nn.MaxPool2d(3, 1, autopad(3)),
                                  ConvBlock(cin, dim3))

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch3(x)
        x2 = self.branch5(x)
        x3 = self.pool(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x


class InceptionNetV1(nn.Module):
    def __init__(self, cin=3):
        super().__init__()

    def forward(self, x):
        return self.features(x)

if __name__ == "__main__":
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    outputs = InceptionBlock(3, 64, (96, 128), (16, 32), 32)(inputs)
    print(outputs.shape)