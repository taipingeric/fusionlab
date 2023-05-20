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

Conv = {
    1:nn.Conv1d,
    2:nn.Conv2d,
    3:nn.Conv3d
}

ConvT = {
    1:nn.ConvTranspose1d,
    2:nn.ConvTranspose2d,
    3:nn.ConvTranspose3d
}

BatchNorm = {
    1:nn.BatchNorm1d,
    2:nn.BatchNorm2d,
    3:nn.BatchNorm3d
}

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