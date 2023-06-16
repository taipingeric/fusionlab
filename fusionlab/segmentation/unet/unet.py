import torch
import torch.nn as nn
from fusionlab.segmentation.base import SegmentationModel
from fusionlab.utils import autopad
from fusionlab.layers import ConvND, MaxPool


class UNet(SegmentationModel):
    def __init__(self, cin, num_cls, base_dim=64, spatial_dims=2):
        """
        Base Unet
        Args:
            cin (int): input channels
            num_cls (int): number of classes
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        stage = 5
        self.num_cls = num_cls
        self.encoder = Encoder(cin, base_dim=base_dim, spatial_dims=spatial_dims)
        self.bridger = Bridger()
        self.decoder = Decoder(cin=base_dim*(2**(stage-1)),
                               base_dim=base_dim*(2**(stage-2)), 
                               spatial_dims=spatial_dims)  # 1024, 512
        self.head = Head(base_dim, num_cls, spatial_dims=spatial_dims)


class Encoder(nn.Module):
    def __init__(self, cin, base_dim, spatial_dims=2):
        """
        UNet Encoder
        Args:
            cin (int): input channels
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        self.pool = MaxPool(spatial_dims, 2, 2)
        self.stage1 = BasicBlock(cin, base_dim, spatial_dims)
        self.stage2 = BasicBlock(base_dim, base_dim * 2, spatial_dims)
        self.stage3 = BasicBlock(base_dim * 2, base_dim * 4, spatial_dims)
        self.stage4 = BasicBlock(base_dim * 4, base_dim * 8, spatial_dims)
        self.stage5 = BasicBlock(base_dim * 8, base_dim * 16, spatial_dims)

    def forward(self, x):
        s1 = self.stage1(x)
        x = self.pool(s1)
        s2 = self.stage2(x)
        x = self.pool(s2)
        s3 = self.stage3(x)
        x = self.pool(s3)
        s4 = self.stage4(x)
        x = self.pool(s4)
        s5 = self.stage5(x)

        return [s1, s2, s3, s4, s5]


class Decoder(nn.Module):
    def __init__(self, cin, base_dim, spatial_dims=2):
        """
        Base UNet decoder
        Args:
            cin (int): input channels
            base_dim (int): output dim of deepest stage output
        """
        super().__init__()
        self.d4 = DecoderBlock(cin, cin//2, base_dim, spatial_dims)
        self.d3 = DecoderBlock(base_dim, cin//4, base_dim//2, spatial_dims)
        self.d2 = DecoderBlock(base_dim//2, cin//8, base_dim//4, spatial_dims)
        self.d1 = DecoderBlock(base_dim//4, cin//16, base_dim//8, spatial_dims)

    def forward(self, x):
        f1, f2, f3, f4, f5 = x
        x = self.d4(f5, f4)
        x = self.d3(x, f3)
        x = self.d2(x, f2)
        x = self.d1(x, f1)
        return x


class Bridger(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        outputs = [nn.Identity()(i) for i in x]
        return outputs


class Head(nn.Sequential):
    def __init__(self, cin, cout, spatial_dims=2):
        """
        Basic conv head
        :param int cin: input channel
        :param int cout: output channel
        """
        conv = ConvND(spatial_dims, cin, cout, 1)
        super().__init__(conv)


class BasicBlock(nn.Sequential):
    def __init__(self, cin, cout, spatial_dims=2):
        conv1 = nn.Sequential(
            ConvND(spatial_dims, cin, cout, 3, 1, autopad(3)),
            nn.ReLU(),
        )
        conv2 = nn.Sequential(
            ConvND(spatial_dims, cout, cout, 3, 1, autopad(3)),
            nn.ReLU(),
        )
        super().__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(self, c1, c2, cout, spatial_dims=2):
        """
        Base Unet decoder block for merging the outputs from 2 stages
        Args:
            c1: input dim of the deeper stage
            c2: input dim of the shallower stage
            cout: output dim of the block
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = BasicBlock(c1 + c2, cout, spatial_dims)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.concat([x1, x2], dim=1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    print("2D UNet")
    H = W = 224
    dim = 64
    inputs = torch.normal(0, 1, (1, 3, H, W))

    encoder = Encoder(3, base_dim=dim)
    outputs = encoder(inputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, dim*(2**i), H//(2**i), W//(2**i)]

    bridger = Bridger()
    outputs = bridger(outputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, dim * (2 ** i), H // (2 ** i), W // (2 ** i)]

    features = [torch.normal(0, 1, (1, dim * (2 ** i), H // (2 ** i), W // (2 ** i))) for i in range(5)]
    decoder = Decoder(1024, 512)
    outputs = decoder(features)
    assert list(outputs.shape) == [1, 64, H, W]

    head = Head(64, 10)
    outputs = head(outputs)
    assert list(outputs.shape) == [1, 10, H, W]

    unet = UNet(3, 10)
    outputs = unet(inputs)
    assert list(outputs.shape) == [1, 10, H, W]

    print("1D UNet")
    L = 224
    dim = 64
    inputs = torch.rand(1, 3, L)
    unet = UNet(3, 10, spatial_dims=1)
    outputs = unet(inputs)
    assert list(outputs.shape) == [1, 10, L]

    print("3D UNet")
    D = H = W = 64
    dim = 32
    inputs = torch.rand(1, 3, D, H, W)
    unet = UNet(3, 10, spatial_dims=3)
    outputs = unet(inputs)
    assert list(outputs.shape) == [1, 10, D, H, W]