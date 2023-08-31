from typing import Sequence, Union
import torch
import torch.nn as nn
import numpy as np

from fusionlab.layers import Rearrange, ConvND
from fusionlab.utils import make_ntuple, trunc_normal_

EMBEDDING_TYPES = ["conv", "fc"]

class PatchEmbedding(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[int, Sequence[int]],
        patch_size: Union[int, Sequence[int]],
        hidden_size: int,
        pos_embed_type: str = 'conv',
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed_type: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """

        super().__init__()
        assert pos_embed_type in EMBEDDING_TYPES, f"pos_embed_type must be in {EMBEDDING_TYPES}"
        self.pos_embed_type = pos_embed_type

        img_sizes = make_ntuple(img_size, spatial_dims)
        patch_sizes = make_ntuple(patch_size, spatial_dims)
        for m, p in zip(img_sizes, patch_sizes):
            if self.pos_embed_type == "fc" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for fc embedding type.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_sizes, patch_sizes)])
        self.patch_dim = int(in_channels * np.prod(patch_sizes))

        self.patch_embeddings: nn.Module
        if self.pos_embed_type == "conv":
            self.patch_embeddings = nn.Sequential(
                ConvND(
                    spatial_dims, 
                    in_channels, 
                    hidden_size, 
                    kernel_size=patch_size, 
                    stride=patch_size
                ),
                nn.Flatten(2),
                Rearrange('b d n -> b n d'),
            )
            # self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
            #     in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            # )
        elif self.pos_embed_type == "fc":
            # for 3d: "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_sizes)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), 
                nn.Linear(self.patch_dim, hidden_size),
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embeddings(x)
        # if self.pos_embed_type == "conv":
        #     x = x.flatten(2).transpose(-1, -2) # (b c w h) -> (b c wh)  -> (b wh c)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    # 2D
    inputs = torch.randn(1, 3, 224, 224)
    l = PatchEmbedding(3, 224, 16, 768, pos_embed_type='conv')
    outputs = l(inputs)
    print(outputs.shape)

    inputs = torch.randn(1, 3, 224, 224)
    l = PatchEmbedding(3, 224, 16, 768, pos_embed_type='fc')
    print(l)
    outputs = l(inputs)
    print(outputs.shape)

    # 1D
    inputs = torch.randn(1, 3, 224)
    l = PatchEmbedding(3, 224, 16, 768, pos_embed_type='conv', spatial_dims=1)
    outputs = l(inputs)
    print(outputs.shape)

    inputs = torch.randn(1, 3, 224)
    l = PatchEmbedding(3, 224, 16, 768, pos_embed_type='fc', spatial_dims=1)
    outputs = l(inputs)
    print(outputs.shape)

    # 3D
    inputs = torch.randn(1, 3, 112, 112, 112)
    l = PatchEmbedding(3, 112, 16, 768, pos_embed_type='conv', spatial_dims=3)
    outputs = l(inputs)
    print(outputs.shape)

    inputs = torch.randn(1, 3, 112, 112, 112)
    l = PatchEmbedding(3, 112, 16, 768, pos_embed_type='fc', spatial_dims=3)
    outputs = l(inputs)
    print(outputs.shape)