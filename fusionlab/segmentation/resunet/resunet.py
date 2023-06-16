import torch
import torch.nn as nn
from fusionlab.segmentation.base import SegmentationModel
from fusionlab.utils import autopad
from fusionlab.layers.factories import ConvND, ConvT, BatchNorm



class ResUNet(SegmentationModel):
    def __init__(self, cin, num_cls, base_dim, spatial_dims=2):
        super().__init__()
        self.num_cls = num_cls
        self.encoder = Encoder(cin, base_dim, spatial_dims)
        self.bridger = Bridger()
        self.decoder = Decoder(base_dim, spatial_dims)
        self.head = Head(base_dim, num_cls, spatial_dims)


class Encoder(nn.Module):
    def __init__(self, cin, base_dim, spatial_dims=2):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(4)]
        self.stem = Stem(cin, dims[0], spatial_dims)
        self.stage1 = ResConv(dims[0], dims[1], spatial_dims, stride=2)
        self.stage2 = ResConv(dims[1], dims[2], spatial_dims, stride=2)
        self.stage3 = ResConv(dims[2], dims[3], spatial_dims, stride=2)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return [s0, s1, s2, s3]


class Decoder(nn.Module):
    def __init__(self, base_dim, spatial_dims=2):
        """
        Base UNet decoder
        Args:
            base_dim (int): output dim of deepest stage output or input channels
        """
        super().__init__()
        dims = [base_dim*(2**i) for i in range(4)]
        self.d3 = DecoderBlock(dims[3], dims[2], spatial_dims)
        self.d2 = DecoderBlock(dims[2], dims[1], spatial_dims)
        self.d1 = DecoderBlock(dims[1], dims[0], spatial_dims)

    def forward(self, x):
        s0, s1, s2, s3 = x

        x = self.d3(s3, s2)
        x = self.d2(x, s1)
        x = self.d1(x, s0)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, spatial_dims=2):
        super().__init__()
        self.upsample = ConvT(spatial_dims, cin, cout, 2, stride=2)
        self.conv = ResConv(cout*2, cout, spatial_dims, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Bridger(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        outputs = [nn.Identity()(i) for i in x]
        return outputs


class Stem(nn.Module):
    def __init__(self, cin, cout, spatial_dims=2):
        super().__init__()
        self.conv = nn.Sequential(
            ConvND(spatial_dims, cin, cout, 3, padding=autopad(3)),
            BatchNorm(spatial_dims, cout),
            nn.ReLU(),
            ConvND(spatial_dims, cout, cout, 3, padding=autopad(3)),
        )
        self.skip = nn.Sequential(
            ConvND(spatial_dims, cin, cout, 3, padding=autopad(3)),
        )

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class ResConv(nn.Module):
    def __init__(self, cin, cout, spatial_dims=2, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            BatchNorm(spatial_dims, cin),
            nn.ReLU(),
            ConvND(spatial_dims, cin, cout, 3, stride, padding=autopad(3)),
            BatchNorm(spatial_dims, cout),
            nn.ReLU(),
            ConvND(spatial_dims, cout, cout, 3, padding=autopad(3)),
        )
        self.skip = nn.Sequential(
            ConvND(spatial_dims, cin, cout, 3, stride=stride, padding=autopad(3)),
            BatchNorm(spatial_dims, cout),
        )

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class Head(nn.Sequential):
    def __init__(self, cin, cout, spatial_dims):
        """
        Basic conv head
        :param int cin: input channel
        :param int cout: output channel
        """
        conv = ConvND(spatial_dims, cin, cout, 1)
        super().__init__(conv)

if __name__ == '__main__':
    H = W = 224
    cout = 64
    inputs = torch.normal(0, 1, (1, 3, H, W))

    model = ResUNet(3, 100, cout)
    output = model(inputs)
    print(output.shape)

    dblock = DecoderBlock(64, 128)
    inputs2 = torch.normal(0, 1, (1, 128, H, W))
    inputs1 = torch.normal(0, 1, (1, 64, H//2, W//2))
    outputs = dblock(inputs1, inputs2)
    print(outputs.shape)

    encoder = Encoder(3, cout)
    outputs = encoder(inputs)
    for o in outputs:
        print(o.shape)

    decoder = Decoder(cout)
    outputs = decoder(outputs)
    print("Encoder + Decoder ", outputs.shape)

    stem = Stem(3, cout)
    outputs = stem(inputs)
    print(outputs.shape)
    assert list(outputs.shape) == [1, cout, H, W]

    resconv = ResConv(3, cout, stride=1)
    outputs = resconv(inputs)
    print(outputs.shape)
    assert list(outputs.shape) == [1, cout, H, W]

    resconv = ResConv(3, cout, stride=2)
    outputs = resconv(inputs)
    print(outputs.shape)
    assert list(outputs.shape) == [1, cout, H//2, W//2]


    print("3D ResUNet")
    D = H = W = 64
    cout = 32
    inputs = torch.rand(1, 3, D, H, W)

    model = ResUNet(3, 100, cout, spatial_dims=3)
    output = model(inputs)
    print(output.shape)

    print("1D ResUNet")
    L = 64
    cout = 32
    inputs = torch.rand(1, 3, L)

    model = ResUNet(3, 100, cout, spatial_dims=1)
    output = model(inputs)
    print(output.shape)
