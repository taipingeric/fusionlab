import torch
import torch.nn as nn
from fusionlab.layers.factories import ConvND, MaxPool

# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
class AlexNet(nn.Module):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvND(spatial_dims, cin, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=3, stride=2),
            ConvND(spatial_dims, 64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=3, stride=2),
            ConvND(spatial_dims, 192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool(spatial_dims, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)



if __name__ == '__main__':
    img_size = 224
    inputs = torch.normal(0, 1, (1, 3, img_size, img_size))
    output = AlexNet(3, spatial_dims=2)(inputs)
    print(output.shape)
    assert list(output.shape) == [1, 256, 6, 6]

    inputs = torch.normal(0, 1, (1, 3, img_size))
    output = AlexNet(3, spatial_dims=1)(inputs)
    print(output.shape)
    assert list(output.shape) == [1, 256, 6]

    img_size = 128
    inputs = torch.normal(0, 1, (1, 3, img_size, img_size, img_size))
    output = AlexNet(3, spatial_dims=3)(inputs)
    print(output.shape)
    assert list(output.shape) == [1, 256, 3, 3, 3]
