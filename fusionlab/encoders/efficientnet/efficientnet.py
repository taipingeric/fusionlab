'''
Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py 
'''

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import StochasticDepth

from dataclasses import dataclass
from typing import Sequence, Optional, Callable, List
from functools import partial
import math
import copy

from fusionlab.layers import ConvNormAct, BatchNorm, SEModule

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    source: https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py#L76
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
        spatial_dims: int = 2,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        spatial_dims: int=2,
        se_layer: Callable[..., nn.Module] = SEModule,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        act_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormAct(
                    spatial_dims,
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormAct(
                spatial_dims,
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, 
                               squeeze_channels, 
                               act_layer=partial(nn.SiLU, inplace=True),
                               spatial_dims=spatial_dims)
        )
        
        # project
        layers.append(
            ConvNormAct(
                spatial_dims,
                expanded_channels, 
                cnf.out_channels, 
                kernel_size=1, 
                norm_layer=norm_layer, 
                act_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig],
        cin: int = 3,
        stochastic_depth_prob: float = 0.2,
        last_channel: Optional[int] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spatial_dims: int = 2,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = BatchNorm

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormAct(
                spatial_dims,
                cin,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                act_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer, spatial_dims))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            ConvNormAct(
                spatial_dims,
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                act_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)

        # weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)

def _build_efficient_cfg(width_mult, depth_mult):
    mb_cfg = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    return [
        mb_cfg(1, 3, 1, 32, 16, 1),
        mb_cfg(6, 3, 2, 16, 24, 2),
        mb_cfg(6, 5, 2, 24, 40, 2),
        mb_cfg(6, 3, 2, 40, 80, 3),
        mb_cfg(6, 5, 1, 80, 112, 3),
        mb_cfg(6, 5, 2, 112, 192, 4),
        mb_cfg(6, 3, 1, 192, 320, 1),
    ]

class EfficientNetB0(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.0, 1.0)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.2, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB1(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.0, 1.1)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.2, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB2(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.1, 1.2)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.3, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB3(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.2, 1.4)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.3, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB4(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.4, 1.8)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.4, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB5(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.6, 2.2)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.4, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB6(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(1.8, 2.6)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.5, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )

class EfficientNetB7(EfficientNet):
    def __init__(self, spatial_dims=2, cin=3):
        config = _build_efficient_cfg(2.0, 3.1)
        last_channel = None
        super().__init__(
            inverted_residual_setting=config, 
            cin=cin,
            stochastic_depth_prob=0.5, 
            last_channel=last_channel, 
            spatial_dims=spatial_dims
        )


if __name__ == '__main__':
    print('efficientnet.')
    model = EfficientNetB0()
    inputs = torch.randn(1, 3, 224, 224)
    outputs = model(inputs)
    print(outputs.shape)


