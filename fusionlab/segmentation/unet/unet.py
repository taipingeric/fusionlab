import torch
import torch.nn as nn
from fusionlab.segmentation.base import SegmentationModel
from fusionlab.utils import autopad


class UNet(SegmentationModel):
    def __init__(self, cin):
        super().__init__()
        self.encoder = UNetEncoder(cin, base_dim=16)


class UNetEncoder(nn.Module):
    def __init__(self, cin, base_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2, 1, autopad(2))
        self.stage1 = UNetStage(cin, base_dim)
        self.stage2 = UNetStage(base_dim, base_dim*2)
        self.stage3 = UNetStage(base_dim*2, base_dim*4)

    def forward(self, x):
        s1 = self.stage1(x)
        x = self.pool(s1)
        s2 = self.stage2(x)
        x = self.pool(s2)
        s3 = self.stage3(x)

        return [s1, s2, s3]


class UNetStage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, autopad(3)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, autopad(3)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    encoder = UNetEncoder(3, base_dim=16)
    outputs = encoder(inputs)
    for o in outputs:
        print(o.shape)
