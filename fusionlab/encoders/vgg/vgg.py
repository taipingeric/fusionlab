import torch
import torch.nn as nn
from fusionlab.layers import ConvND, MaxPool

# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
class VGG16(nn.Module):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            ConvND(spatial_dims, cin, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class VGG19(nn.Module):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            ConvND(spatial_dims, cin, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),

            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)




if __name__ == '__main__':
    # VGG16
    inputs = torch.normal(0, 1, (1, 3, 224))
    output = VGG16(cin=3, spatial_dims=1)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7]

    # VGG19
    inputs = torch.normal(0, 1, (1, 3, 224))
    output = VGG19(cin=3, spatial_dims=1)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7]

    # VGG16
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG16(cin=3,  spatial_dims=2)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7, 7]

    # VGG19
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG19(cin=3,  spatial_dims=2)(inputs)
    shape = list(output.shape)
    assert shape[2:] == [7, 7]