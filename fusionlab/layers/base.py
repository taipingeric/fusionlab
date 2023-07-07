import torch.nn as nn
from typing import Union, Sequence, Optional, Callable

from fusionlab.layers import ConvND, BatchNorm

class ConvNormAct(nn.Module):
    def __init__(self, 
        spatial_dims: int, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        padding: Union[Sequence[int], str] = 0,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = BatchNorm,
        act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        padding_mode: str = 'zeros',
        inplace: Optional[bool] = bool,
    ):
        super().__init__()
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
