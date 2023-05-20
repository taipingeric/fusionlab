from torch import nn
from typing import Union, Sequence

class ConvND:
    """
    Factory class for creating convolutional layers. 
    This class is used to create convolutional layers with the same configuration.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        in_channels (int): number of channels in the input data.
        out_channels (int): number of channels produced by the convolution.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple, optional): stride of the convolution. Default: 1
        padding (int or tuple, optional): zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): spacing between kernel elements. Default: 1
        groups (int, optional): number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): whether to add a bias to the convolution. Default: True
        padding_mode (str, optional): type of padding. Default: 'zeros'

    """
    def __new__(cls, 
                spatial_dims, 
                in_channels: int,
                out_channels: int,
                kernel_size: Union[Sequence[int], int],
                stride: Union[Sequence[int], int] = 1,
                padding: Union[Sequence[int], str] = 0,
                dilation: Union[Sequence[int], int] = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'`spatial_dims` must be 1, 2, or 3, got {spatial_dims}')
        conv_type = getattr(nn, f'Conv{spatial_dims}d')
        return conv_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

class Upsample:
    def __new__(cls,
                spatial_dims,
                size=None, 
                scale_factor=None, 
                mode=None, 
                align_corners=None):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'`spatial_dims` must be 1, 2, or 3, got {spatial_dims}')
        if not mode:
            mode_map = {
                1: 'linear',
                2: 'bilinear',
                3: 'trilinear'
            }
            mode = mode_map[spatial_dims]
            align_corners=False
        return nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

class ConvT:
    """
    Factory class for creating transposed convolutional layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        in_channels (int): number of channels in the input data.
        out_channels (int): number of channels produced by the convolution.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple, optional): stride of the convolution. Default: 1
        padding (int or tuple, optional): zero-padding added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): additional size added to one side of each dimension in the output shape. Default: 0
        groups (int, optional): number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): whether to add a bias to the convolution. Default: True
        dilation (int or tuple, optional): spacing between kernel elements. Default: 1
        padding_mode (str, optional): type of padding. Default: 'zeros'

    """
    def __new__(cls, 
                spatial_dims, 
                in_channels: int,
                out_channels: int,
                kernel_size: Union[Sequence[int], int],
                stride: Union[Sequence[int], int] = 1,
                padding: Union[Sequence[int], str] = 0,
                output_padding: Union[Sequence[int], str] = 0,
                groups: int = 1,
                bias: bool = True,
                dilation: Union[Sequence[int], int] = 1,
                padding_mode: str = 'zeros'):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'`spatial_dims` must be 1, 2, or 3, got {spatial_dims}')
        conv_type = getattr(nn, f'ConvTranspose{spatial_dims}d')
        return conv_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

class BatchNorm:
    def __new__(cls, spatial_dims, 
                num_features: int,
                eps: float = 1e-5,
                momentum: float = 0.1,
                affine: bool = True,
                track_running_stats: bool = True):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'`spatial_dims` must be 1, 2, or 3, got {spatial_dims}')
        bn_type = getattr(nn, f'BatchNorm{spatial_dims}d')
        return bn_type(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

MaxPool = {
    1:nn.MaxPool1d,
    2:nn.MaxPool2d,
    3:nn.MaxPool3d
}

AdaptiveMaxPool = {
    1:nn.AdaptiveMaxPool1d,
    2:nn.AdaptiveMaxPool2d,
    3:nn.AdaptiveMaxPool3d
}

AvgPool = {
    1:nn.AvgPool1d,
    2:nn.AvgPool2d,
    3:nn.AvgPool3d
}

AdaptiveAvgPool = {
    1:nn.AdaptiveAvgPool1d,
    2:nn.AdaptiveAvgPool2d,
    3:nn.AdaptiveAvgPool3d
}

ReplicationPad = {
    1:nn.ReplicationPad1d,
    2:nn.ReplicationPad2d,
    3:nn.ReplicationPad3d
}

ConstantPad = {
    1:nn.ConstantPad1d,
    2:nn.ConstantPad2d,
    3:nn.ConstantPad3d
}