from typing import Union, Sequence
import numpy as np
import torch
import torch.nn as nn
from fusionlab.encoders import ViT
from fusionlab.layers import InstanceNorm, ConvND, ConvT
from fusionlab.utils import make_ntuple

class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvND(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size,
                stride,
            ),
            InstanceNorm(spatial_dims, out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            ConvND(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
            ),
            InstanceNorm(spatial_dims, out_channels),
        )
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.downsample = in_channels != out_channels

        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.downsample_block = nn.Sequential(
                ConvND(
                    spatial_dims,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                InstanceNorm(spatial_dims, out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.downsample:
            residual = self.downsample_block(residual)
        out += residual
        out = self.act(out)
        return out

class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = ConvT(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvT(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=upsample_kernel_size,
                    stride=upsample_stride,
                ),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            for _ in range(num_layer)
        ])

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int], # Sequence[int] | int,
        upsample_kernel_size: Union[Sequence[int], int], # Sequence[int] | int,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = ConvT(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
        )
        self.conv_block = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels*2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, x, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out

# TODO: test this module
class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

    source code: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "fc",
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop in ViT.
            spatial_dims: number of spatial dimensions.
        """

        super().__init__()
        self.num_layers = 12
        self.patch_size = make_ntuple(16, spatial_dims)
        img_size = make_ntuple(img_size, spatial_dims)
        self.feat_size = tuple([img_size[i] // self.patch_size[i] for i in range(spatial_dims)])
        self.spatial_dims = spatial_dims
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
        )
        self.out = ConvND(
            spatial_dims,
            feature_size,
            out_channels,
            kernel_size=1
        )

    def proj_feat(self, x, hidden_size, feat_size):
        target_size = [x.size(0)] + list(feat_size) + [hidden_size]
        x = x.view(*target_size)
        # swap the spatial and feature dimensions
        permute_order = [0] + [self.spatial_dims+1] + [i+1 for i in range(self.spatial_dims)]
        x = x.permute(*permute_order).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in, return_features=True)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
    

if __name__ == '__main__':
    # 2D
    inputs = torch.randn(1, 3, 128, 128)
    model = UNETR(
        in_channels=3,
        out_channels=4,
        img_size=128,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        spatial_dims=2,
    )
    outputs = model(inputs)
    print(outputs.shape)

    # 3D
    inputs = torch.randn(1, 3, 64, 64, 64)
    model = UNETR(
        in_channels=3,
        out_channels=4,
        img_size=64,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        spatial_dims=3,
    )
    outputs = model(inputs)
    print(outputs.shape)

    # 1D
    inputs = torch.randn(1, 3, 64)
    model = UNETR(
        in_channels=3,
        out_channels=4,
        img_size=64,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        spatial_dims=1,
    )
    outputs = model(inputs)
    print(outputs.shape)