import torch
from torch import nn
from fusionlab.segmentation.base import SegmentationModel
from fusionlab.utils import autopad


class UNet2plus(SegmentationModel):
    def __init__(self, cin, num_cls, base_dim):
        super().__init__()
        self.encoder = Encoder(cin, base_dim)
        self.bridger = Bridger()
        self.decoder = Decoder(base_dim)
        self.head = Head(base_dim, num_cls)


class BasicBlock(nn.Sequential):
    def __init__(self, cin, cout):
        conv1 = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, autopad(3)),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, autopad(3)),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )
        super().__init__(conv1, conv2)


class Encoder(nn.Module):
    def __init__(self, cin, base_dim):
        """
        UNet Encoder
        Args:
            cin (int): input channels
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = BasicBlock(cin, base_dim)
        self.conv1_0 = BasicBlock(base_dim, base_dim * 2)
        self.conv2_0 = BasicBlock(base_dim * 2, base_dim * 4)
        self.conv3_0 = BasicBlock(base_dim * 4, base_dim * 8)
        self.conv4_0 = BasicBlock(base_dim * 8, base_dim * 16)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return [x0_0, x1_0, x2_0, x3_0, x4_0]


class Bridger(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [nn.Identity()(i) for i in x]


class Decoder(nn.Module):
    def __init__(self, base_dim):
        super().__init__()
        dims = [base_dim*(2**i) for i in range(5)]  # [base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*16]
        self.conv0_1 = BasicBlock(dims[0] + dims[1], dims[0])
        self.conv1_1 = BasicBlock(dims[1] + dims[2], dims[1])
        self.conv2_1 = BasicBlock(dims[2] + dims[3], dims[2])
        self.conv3_1 = BasicBlock(dims[3] + dims[4], dims[3])

        self.conv0_2 = BasicBlock(dims[0] * 2 + dims[1], dims[0])
        self.conv1_2 = BasicBlock(dims[1] * 2 + dims[2], dims[1])
        self.conv2_2 = BasicBlock(dims[2] * 2 + dims[3], dims[2])

        self.conv0_3 = BasicBlock(dims[0] * 3 + dims[1], dims[0])
        self.conv1_3 = BasicBlock(dims[1] * 3 + dims[2], dims[1])

        self.conv0_4 = BasicBlock(dims[0] * 4 + dims[1], dims[0])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = x

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return x0_4


class Head(nn.Sequential):
    def __init__(self, cin, cout):
        """
        Basic Identity
        :param int cin: input channel
        :param int cout: output channel
        """
        conv = nn.Conv2d(cin, cout, 1)
        super().__init__(conv)


if __name__ == '__main__':
    H = W = 224
    dim = 32
    inputs = torch.normal(0, 1, (1, 3, H, W))

    encoder = Encoder(3, base_dim=dim)
    outputs = encoder(inputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, dim * (2 ** i), H // (2 ** i), W // (2 ** i)]

    bridger = Bridger()
    outputs = bridger(outputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, dim * (2 ** i), H // (2 ** i), W // (2 ** i)]

    features = [torch.normal(0, 1, (1, dim * (2 ** i), H // (2 ** i), W // (2 ** i))) for i in range(5)]
    decoder = Decoder(dim)
    outputs = decoder(features)
    assert list(outputs.shape) == [1, dim, H, W]

    head = Head(dim, 10)
    outputs = head(outputs)
    assert list(outputs.shape) == [1, 10, H, W]

    unet = UNet2plus(3, 10, dim)
    outputs = unet(inputs)
    assert list(outputs.shape) == [1, 10, H, W]
