from typing import Sequence
import torch
import torch.nn as nn
from fusionlab.layers import (
    SRAttention,
    DropPath,
)

class PatchEmbed(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            out_channels=32, 
            patch_size=7, 
            stride=4
        ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, 
            out_channels, 
            patch_size, stride, 
            patch_size//2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class MiTBlock(nn.Module):
    def __init__(self, dim, head, spatio_reduction_ratio=1, drop_path_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(dim, head, spatio_reduction_ratio)
        self.drop_path = DropPath(drop_path_rate, "row") if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: torch.Tensor, h, w) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.fc2 = nn.Linear(c2, c1)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor, h, w) -> torch.Tensor:
        x = self.fc1(x)
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MiT(nn.Module):
    """
    Mix Transformer

    source code: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py
    """
    def __init__(
            self,
            in_channels: int = 3, 
            embed_dims: Sequence[int] = [32, 64, 160, 256],
            depths: Sequence[int] = [2, 2, 2, 2]
        ):
        super().__init__()
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(in_channels, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([MiTBlock(embed_dims[0], 1, 8, drop_path_rate[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([MiTBlock(embed_dims[1], 2, 4, drop_path_rate[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([MiTBlock(embed_dims[2], 5, 2, drop_path_rate[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([MiTBlock(embed_dims[3], 8, 1, drop_path_rate[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])


    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        bs = x.shape[0]

        # stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x1 = self.norm1(x).reshape(bs, h, w, -1).permute(0, 3, 1, 2)

        # stage 2
        x, h, w = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, h, w)
        x2 = self.norm2(x).reshape(bs, h, w, -1).permute(0, 3, 1, 2)

        # stage 3
        x, h, w = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, h, w)
        x3 = self.norm3(x).reshape(bs, h, w, -1).permute(0, 3, 1, 2)

        # stage 4
        x, h, w = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, h, w)
        x4 = self.norm4(x).reshape(bs, h, w, -1).permute(0, 3, 1, 2)

        if return_features:
            return x4, [x1, x2, x3, x4]
        else:
            return x4

class MiTB0(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [32, 64, 160, 256], [2, 2, 2, 2])

class MiTB1(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [64, 128, 320, 512], [2, 2, 2, 2])

class MiTB2(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [64, 128, 320, 512], [3, 4, 6, 3])

class MiTB3(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [64, 128, 320, 512], [3, 4, 18, 3])

class MiTB4(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [64, 128, 320, 512], [3, 8, 27, 3])

class MiTB5(MiT):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, [64, 128, 320, 512], [3, 6, 40, 3])

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 128, 128)
    for i in range(6):
        # model = MiT(in_channels=3)
        model = eval(f'MiTB{i}')(in_channels=3)
        outputs = model(inputs)
        print(outputs.shape)