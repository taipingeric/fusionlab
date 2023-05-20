import torch
import torch.nn as nn
from fusionlab.layers.factories import Conv

class SEModule(nn.Module):
    def __init__(self, cin, ratio=16, spatial_dims=2):
        super().__init__()
        cout = int(cin / ratio)
        self.gate = nn.Sequential(
            Conv(spatial_dims,cin, cout, kernel_size=1),
            nn.ReLU(inplace=True),
            Conv(spatial_dims,cout, cin, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        x = inputs.mean((-2, -1), keepdim=True)
        x = self.gate(x)
        return inputs * x


if __name__ == '__main__':
    inputs = torch.normal(0, 1, (1, 256, 224, 224))
    layer = SEModule(256)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 224, 224]
