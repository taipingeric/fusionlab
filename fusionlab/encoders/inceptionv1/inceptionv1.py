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
        self.stem = nn.Sequential(
            ConvBlock(cin, 64, 7, stride=2),
            nn.MaxPool2d(3, 2, padding=autopad(3)),
            ConvBlock(64, 192, 3),
            nn.MaxPool2d(3, 2, padding=autopad(3)),
        )
        self.incept3a = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.incept3b = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = nn.MaxPool2d(3, 2, padding=autopad(3))
        self.incept4a = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.incept4b = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.incept4c = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.incept4d = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.incept4e = InceptionBlock(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = nn.MaxPool2d(3, 2, padding=autopad(3))
        self.incept5a = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.incept5b = InceptionBlock(832, 384, (192, 384), (48, 128), 128)

    def forward(self, x):
        x = self.stem(x)
        x = self.incept3a(x)
        x = self.incept3b(x)
        x = self.pool3(x)
        x = self.incept4a(x)
        x = self.incept4b(x)
        x = self.incept4c(x)
        x = self.incept4d(x)
        x = self.incept4e(x)
        x = self.pool4(x)
        x = self.incept5a(x)
        x = self.incept5b(x)
        return x

if __name__ == "__main__":
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    outputs = InceptionBlock(3, 64, (96, 128), (16, 32), 32)(inputs)
    print(outputs.shape)
    assert list(outputs.shape) == [1, 256, 224, 224]

    outputs = InceptionNetV1()(inputs)
    assert list(outputs.shape) == [1, 1024, 7, 7]
