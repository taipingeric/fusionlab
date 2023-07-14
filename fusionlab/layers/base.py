import torch.nn as nn
from typing import Union, Sequence, Optional, Callable

from fusionlab.layers import ConvND, BatchNorm
from fusionlab.utils import make_ntuple

class ConvNormAct(nn.Module):
    '''
    ref: 
    https://pytorch.org/vision/main/generated/torchvision.ops.Conv2dNormActivation.html
    https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L68

    Convolution + Normalization + Activation

    Args:
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of channels of the input image.
        out_channels (int): number of channels of the output image.
        kernel_size (Union[Sequence[int], int]): size of the convolving kernel.
        stride (Union[Sequence[int], int], optional): stride of the convolution. Default: 1
        padding (Union[Sequence[int], str], optional): Padding added to all four sides of the input. Default: None, 
            in which case it will be calculated as padding = (kernel_size - 1) // 2 * dilation
        dilation (Union[Sequence[int], int], optional): spacing between kernel elements. Default: 1
        groups (int, optional): number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if norm_layer is None.
        norm_layer (Optional[Callable[..., nn.Module]], optional): normalization layer. Default: BatchNorm
        act_layer (Optional[Callable[..., nn.Module]], optional): activation layer. Default: nn.ReLU
        padding_mode (str, optional): mode of padding. Default: 'zeros'
        inplace (Optional[bool], optional): Parameter for the activation layer, 
            which can optionally do the operation in-place. Default True

    '''
    def __init__(self, 
        spatial_dims: int, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        padding: Union[Sequence[int], str] = None,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = BatchNorm,
        act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        padding_mode: str = 'zeros',
        inplace: Optional[bool] = bool,
    ):
        super().__init__()
        # padding 
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = spatial_dims
                kernel_size = make_ntuple(kernel_size, _conv_dim)
                dilation = make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        # bias
        if bias is None:
            bias = norm_layer is None

        self.conv = ConvND(
            spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias, 
            padding_mode
        )
        self.norm = norm_layer(spatial_dims, out_channels)
        params = {} if inplace is None else {"inplace": inplace}
        if act_layer is None:
            act_layer = nn.Identity
        self.act = act_layer(**params)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
