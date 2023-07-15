from typing import Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from fusionlab.layers import ConvND, BatchNorm, MaxPool

# source code: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

__all__ = [
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNetV1",
]

class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spatial_dims=2,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvND(
            spatial_dims,
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(spatial_dims, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvND(
            spatial_dims,
            planes,
            planes,
            kernel_size=3,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(spatial_dims, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spatial_dims=2,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm
            # norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvND(
            spatial_dims,
            inplanes,
            width,
            kernel_size=1,
            bias=False
        )
        self.bn1 = norm_layer(spatial_dims, width)
        self.conv2 = ConvND(
            spatial_dims,
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(spatial_dims, width)
        self.conv3 = ConvND(
            spatial_dims,
            width,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_layer(spatial_dims, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Stem(nn.Module):
    def __init__(
        self, 
        cin: int,
        inplanes: int, 
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spatial_dims=2
    ):
        super().__init__()
        self.conv1 = ConvND(spatial_dims, cin, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(spatial_dims, inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = MaxPool(spatial_dims, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        cin=3,
        spatial_dims=2,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm
        self.zero_init_residual = zero_init_residual
        self._norm_layer = norm_layer
        self.spatial_dims = spatial_dims
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Stem(cin, self.inplanes, norm_layer, spatial_dims=spatial_dims)
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.apply(self._init_weights)
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ConvND(self.spatial_dims, 
                       self.inplanes, planes * block.expansion, 
                       kernel_size=1, 
                       stride=stride, 
                       bias=False
                ),
                norm_layer(self.spatial_dims, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                spatial_dims=self.spatial_dims
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    spatial_dims=self.spatial_dims,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward_features(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


class ResNet18(ResNet):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__(BasicBlock, [2, 2, 2, 2], cin=cin, spatial_dims=spatial_dims)

class ResNet34(ResNet):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__(BasicBlock, [3, 4, 6, 3], cin=cin, spatial_dims=spatial_dims)

class ResNet50(ResNet):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__(Bottleneck, [3, 4, 6, 3], cin=cin, spatial_dims=spatial_dims)

class ResNet101(ResNet):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__(Bottleneck, [3, 4, 23, 3], cin=cin, spatial_dims=spatial_dims)

class ResNet152(ResNet):
    def __init__(self, cin=3, spatial_dims=2):
        super().__init__(Bottleneck, [3, 8, 36, 3], cin=cin, spatial_dims=spatial_dims)

ResNetV1 = ResNet

if __name__ == "__main__":
    print('ResNetV2')

    names = ['18', '34', '50', '101', '152']
    blocks = [BasicBlock, BasicBlock, Bottleneck, Bottleneck, Bottleneck]
    layers = [
        [2, 2, 2, 2], 
        [3, 4, 6, 3], 
        [3, 4, 6, 3], 
        [3, 4, 23, 3], 
        [3, 8, 36, 3]
    ]
    # ResNet from torchvision
    params = [
        11176512,
        21284672,
        23508032,
        42500160,
        58143808,
    ]
    output_dims = [512, 512, 2048, 2048, 2048]

    import torchinfo
    for name, block, layer, param, dim in zip(names, blocks, layers, params, output_dims):
        print(f'ResNet{name}')
        inputs = torch.randn(1, 3, 224, 224)
        model = eval(f'ResNet{name}')() #ResNet(block, layer)
        outputs = model(inputs)
        print(outputs.shape)
        logs = torchinfo.summary(model, inputs.shape, verbose=0)
        assert outputs.shape == (1, dim, 7, 7)
        assert logs.total_params == param