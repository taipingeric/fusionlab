from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusionlab.encoders import (
    MiT,
    MiTB0,
    MiTB1,
    MiTB2,
    MiTB3,
    MiTB4,
    MiTB5,
)

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2) # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))

class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg

class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers<https://arxiv.org/abs/2105.15203>

    source code: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/segformer.py

    Args:

        num_classes (int): number of classes to segment
        mit_encoder_type (str): type of MiT encoder, one of ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']
    """
    def __init__(
            self, 
            num_classes: int = 6, 
            mit_encoder_type: str = 'B0'
        ):
        super().__init__()
        self.encoder: MiT = eval(f'MiT{mit_encoder_type}')()
        embed_dim = self.encoder.channels[-1]
        self.decode_head = SegFormerHead(
            self.encoder.channels,
            embed_dim, 
            num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, features = self.encoder(inputs, return_features=True)
        x = self.decode_head(features)   # 4x reduction in image size
        x = F.interpolate(x, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        return x
    
if __name__ == '__main__':
    model = SegFormer(num_classes=6)
    x = torch.randn(1, 3, 128, 128)
    outputs = model(x)
    print(outputs.shape)
    
