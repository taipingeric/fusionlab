import torch
import torch.nn as nn


# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
class VGG16(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class VGG19(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)




if __name__ == '__main__':
    # VGG16
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG16(3)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7, 7]

    # VGG19
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG19(3)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7, 7]