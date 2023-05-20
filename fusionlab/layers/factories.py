from typing import Union, Sequence
import torch
from torch import nn

# a general convolution layer for 1, 2, 3 dimensions
class Conv:
    def __new__(cls,
                spatial_dims: int,
                in_channels: int,
                out_channels: int,
                kernel_size: Union [Sequence [int], int],
                stride: Union [Sequence [int], int] = 1,
                padding: Union [ Sequence [int], str] = 0,
                dilation: Union [Sequence [int], int] = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'Conv{spatial_dims}d')
        return conv_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

# a general transposed convolution layer for 1, 2, 3 dimensions
class ConvT:
    def __new__(cls,
                spatial_dims: int,
                in_channels: int,
                out_channels: int,
                kernel_size: Union [Sequence [int], int],
                stride: Union [Sequence [int], int] = 1,
                padding: Union [Sequence [int], int] = 0,
                output_padding: Union [Sequence [int], int] = 0,
                groups: int = 1,
                bias: bool = True,
                dilation: int = 1,
                padding_mode: str = 'zeros'):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'ConvTranspose{spatial_dims}d')
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
            padding_mode=padding_mode)

# a general batch normalization layer for 1, 2, 3 dimensions
class BatchNorm:
    def __new__(cls,
        spatial_dims: int,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'BatchNorm{spatial_dims}d')
        return conv_type(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)

# a general max pooling layer for 1, 2, 3 dimensions
class MaxPool:
    def __new__(cls,
                spatial_dims: int, 
                kernel_size: Union [int, Sequence [int]],
                stride: Union [Sequence [int], int, None] = None,
                padding: Union [Sequence [int], int] = 0,
                dilation: Union [Sequence [int], int] = 1,
                return_indices: bool = False,
                ceil_mode: bool = False):
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'MaxPool{spatial_dims}d')
        return conv_type(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)

# a general average pooling layer for 1, 2, 3 dimensions
class AvgPool:
    def __new__(cls,
                spatial_dims: int, 
                kernel_size: Union [int, Sequence [int]],
                stride: Union [int, Sequence [int], None] = None,
                padding: Union [int, Sequence [int]] = 0,
                ceil_mode: bool = False,
                count_include_pad: bool = True,
                divisor_override: Union [int, None] = None):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'AvgPool{spatial_dims}d')
        return conv_type(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override)

# a general adaptive max pooling layer for 1, 2, 3 dimensions
class AdaptiveMaxPool:
    def __new__(cls,
                spatial_dims: int, 
                output_size: Union [int, Sequence [int]], 
                return_indices: bool = False):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'AdaptiveMaxPool{spatial_dims}d')
        return conv_type(
            output_size=output_size,
            return_indices=return_indices)    

# a general adaptive average pooling layer for 1, 2, 3 dimensions
class AdaptiveAvgPool:
    def __new__(cls,
                spatial_dims: int, 
                output_size: Union [int, Sequence [int]]):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'AdaptiveAvgPool{spatial_dims}d')
        return conv_type(
            output_size=output_size)

# a general replicated padding layer for 1, 2, 3 dimensions
class ReplicationPad:
    def __new__(cls,
                spatial_dims: int,
                padding: Union [int, Sequence [int]]):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'ReplicationPad{spatial_dims}d')
        return conv_type(
            padding=padding)

# a general constant padding layer for 1, 2, 3 dimensions
class ConstantPad:
    def __new__(cls,
                spatial_dims: int,
                padding: Union [int, Sequence [int]],
                value: float):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'ConstantPad{spatial_dims}d')
        return conv_type(
            padding=padding,
            value=value)

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = Conv(spatial_dims=1, in_channels=3, out_channels=2, kernel_size=5) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 2, 12] # check output shape is correct
