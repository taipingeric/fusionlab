import torch
import torch.nn as nn
from fusionlab.layers.factories import ConvND

class SEModule(nn.Module):
    def __init__(self, cin, ratio=16, spatial_dims=2):
        super().__init__()
        cout = int(cin / ratio)
        self.gate = nn.Sequential(
            ConvND(spatial_dims, cin, cout, kernel_size=1),
            nn.ReLU(inplace=True),
            ConvND(spatial_dims, cout, cin, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_dims = spatial_dims

    def forward(self, inputs):
        mean_dims = tuple(range(2, 2 + self.spatial_dims))
        x = inputs.mean(mean_dims, keepdim=True)
        x = self.gate(x)
        return inputs * x


if __name__ == '__main__':
    inputs = torch.normal(0, 1, (1, 256, 16, 16))
    layer = SEModule(256)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 16, 16]

    inputs = torch.normal(0, 1, (1, 256, 16, 16, 16))
    layer = SEModule(256, spatial_dims=3)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 16, 16, 16]
