from typing import Union, Sequence
import torch
from torch import nn

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
            padding_mode=padding_mode)

class Upsample:
    """
    Factory class for creating Upsample layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.
    """
    def __new__(cls,
                spatial_dims: int,
                size: Union[Sequence[int], int, None] = None,
                scale_factor: Union[Sequence[int], int, None] = None,
                mode: str = 'nearest',
                align_corners: Union[bool, None] = None,
                recompute_scale_factor: Union[bool, None] = None):
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
            padding_mode=padding_mode)

class BatchNorm:
    """
    Factory class for creating batch normalization layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """
    def __new__(cls, 
                spatial_dims: int, 
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
            track_running_stats=track_running_stats)

class MaxPool:
    """
    Factory class for creating maximum pooling layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
    """
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
        pool_type=getattr(nn, f'MaxPool{spatial_dims}d')
        return pool_type(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)


class AvgPool:
    """
    Factory class for creating average pooling layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
    (the original version for 2d and 3d has a 'divisor_override' parameter which is neglected here)
    """
    def __new__(cls,
                spatial_dims: int, 
                kernel_size: Union [int, Sequence [int]],
                stride: Union [int, Sequence [int], None] = None,
                padding: Union [int, Sequence [int]] = 0,
                ceil_mode: bool = False,
                count_include_pad: bool = True):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        conv_type=getattr(nn, f'AvgPool{spatial_dims}d')
        return conv_type(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad)


class AdaptiveMaxPool:
    """
    Factory class for creating adaptive max pooling layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        output_size: the target output size :math:`L_{out}`.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool1d. Default: ``False``
    """
    def __new__(cls,
                spatial_dims: int, 
                output_size: Union [int, Sequence [int]], 
                return_indices: bool = False):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        pool_type=getattr(nn, f'AdaptiveMaxPool{spatial_dims}d')
        return pool_type(
            output_size=output_size,
            return_indices=return_indices)    


# a general adaptive average pooling layer for 1, 2, 3 dimensions
class AdaptiveAvgPool:
    """
    Factory class for creating adaptive average pooling layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
    def __new__(cls,
                spatial_dims: int, 
                output_size: Union [int, Sequence [int]]):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        pool_type=getattr(nn, f'AdaptiveAvgPool{spatial_dims}d')
        return pool_type(
            output_size=output_size)

class ReplicationPad:
    """
    Factory class for creating replication padding layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
    """
    def __new__(cls,
                spatial_dims: int,
                padding: Union [int, Sequence [int]]):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        pad_type=getattr(nn, f'ReplicationPad{spatial_dims}d')
        return pad_type(
            padding=padding)


# a general constant padding layer for 1, 2, 3 dimensions
class ConstantPad:
    """
    Factory class for creating adaptive average pooling layers.

    Args:
        spatial_dims (int): number of spatial dimensions of the input data.
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
        value (float): the value of padding
    """
    def __new__(cls,
                spatial_dims: int,
                padding: Union [int, Sequence [int]],
                value: float):
        if spatial_dims not in [1, 2, 3]:
                raise ValueError(f'spatial_dims must be 1, 2, or 3, got {spatial_dims}')
        pad_type=getattr(nn, f'ConstantPad{spatial_dims}d')
        return pad_type(
            padding=padding,
            value=value)


if __name__ == '__main__':

    # Test Code for ConvND
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = ConvND(spatial_dims=1, in_channels=3, out_channels=2, kernel_size=5) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 2, 12] # check output shape is correct

    # Test code for ConvT
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = ConvT(spatial_dims=1, in_channels=3, out_channels=2, kernel_size=5) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 2, 20] # check output shape is correct

    # Test code for Upsample
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = Upsample(spatial_dims=1, scale_factor=2) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 3, 32] # check output shape is correct

    # Test code for BatchNormND
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = BatchNorm(spatial_dims=1, num_features=3) # create model instance
    outputs = layer(inputs) # pass input through model

    # Test code for MaxPool
    for Module in [MaxPool, AvgPool]:
        inputs = torch.randn(1, 3, 16) # create random input tensor
        layer = Module(spatial_dims=1, kernel_size=2) # create model instance
        outputs = layer(inputs) # pass input through model
        print(outputs.shape)
        assert list(outputs.shape) == [1, 3, 8] # check output shape is correct

    # Test code for Pool
    for Module in [AdaptiveMaxPool, AdaptiveAvgPool]:
        inputs = torch.randn(1, 3, 16) # create random input tensor
        layer = Module(spatial_dims=1, output_size=8) # create model instance
        outputs = layer(inputs) # pass input through model
        print(outputs.shape)
        assert list(outputs.shape) == [1, 3, 8] # check output shape is correct

    # Test code for Padding
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = ReplicationPad(spatial_dims=1, padding=2) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 3, 20] # check output shape is correct
    
    layer = ConstantPad(spatial_dims=1, padding=2, value=0) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 3, 20] # check output shape is correct