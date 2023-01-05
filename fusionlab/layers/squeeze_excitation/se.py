import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, cin, ratio=16):
        super().__init__()
        cout = int(cin / ratio)
        self.gate = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cin, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        x = inputs.mean((2, 3), keepdim=True)
        x = self.gate(x)
        return inputs * x


if __name__ == '__main__':
    inputs = torch.normal(0, 1, (1, 256, 224, 224))
    layer = SEModule(256)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 224, 224]
