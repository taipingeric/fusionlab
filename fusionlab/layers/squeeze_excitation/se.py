from typing import Callable
import torch
from torch import Tensor
import torch.nn as nn
from fusionlab.layers import ConvND, AdaptiveAvgPool

class SEModule(nn.Module):
    """
    source: https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L224
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        act_layer: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_layer: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool(spatial_dims, 1)
        self.fc1 = ConvND(spatial_dims, input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = ConvND(spatial_dims, squeeze_channels, input_channels, kernel_size=1)
        self.act_layer = act_layer()
        self.scale_layer = scale_layer()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.act_layer(scale)
        scale = self.fc2(scale)
        return self.scale_layer(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


if __name__ == '__main__':
    print('SEModule')
    inputs = torch.normal(0, 1, (1, 256, 16, 16))
    layer = SEModule(256)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 16, 16]

    inputs = torch.normal(0, 1, (1, 256, 16, 16, 16))
    layer = SEModule(256, spatial_dims=3)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 256, 16, 16, 16]
