'''
Ref: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
'''
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth
from fusionlab.layers import ConvND

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, spatial_dims=2):
        super().__init__()
        self.dwconv = ConvND(spatial_dims, dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, 'N C ... -> N ... C')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, 'N ... C -> N C ...')

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0., 
        layer_scale_init_value=1e-6,
        spatial_dims=2,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            ConvND(spatial_dims, in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    ConvND(spatial_dims, dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                spatial_dims=spatial_dims) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (N, C, H, W).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_first":
            # to channel last
            x = rearrange(x, 'N C ... -> N ... C')
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = rearrange(x, 'N ... C -> N C ...')
            return x
        elif self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class ConvNeXtTiny(ConvNeXt):
    def __init__(self, spatial_dims=2, cin=3):
        super().__init__(depths=[3, 3, 9, 3], 
                         dims=[96, 192, 384, 768],
                         in_chans=cin,
                         spatial_dims=spatial_dims)

class ConvNeXtSmall(ConvNeXt):
    def __init__(self, spatial_dims=2, cin=3):
        super().__init__(depths=[3, 3, 27, 3], 
                         dims=[96, 192, 384, 768],
                         in_chans=cin,
                         spatial_dims=spatial_dims)

class ConvNeXtBase(ConvNeXt):
    def __init__(self, spatial_dims=2, cin=3):
        super().__init__(depths=[3, 3, 27, 3], 
                         dims=[128, 256, 512, 1024],
                         in_chans=cin,
                         spatial_dims=spatial_dims)
        
class ConvNeXtLarge(ConvNeXt):
    def __init__(self, spatial_dims=2, cin=3):
        super().__init__(depths=[3, 3, 27, 3], 
                         dims=[192, 384, 768, 1536],
                         in_chans=cin,
                         spatial_dims=spatial_dims)
        
class ConvNeXtXLarge(ConvNeXt):
    def __init__(self, spatial_dims=2, cin=3):
        super().__init__(depths=[3, 3, 27, 3], 
                         dims=[256, 512, 1024, 2048],
                         in_chans=cin,
                         spatial_dims=spatial_dims)


if __name__ == '__main__':
    print('ConvNeXt')
    for spatial_dims in [1, 2, 3]:
        for i, convnext_type in enumerate(['Tiny', 'Small', 'Base', 'Large', 'XLarge']):
            # print()
            model = eval(f'ConvNeXt{convnext_type}')(spatial_dims=spatial_dims)
            input_size = tuple([1, 3] + [64] * spatial_dims)
            inputs = torch.randn(input_size)
            outputs = model(inputs)
            target_ch = [768, 768, 1024, 1536, 2048]
            assert outputs.shape == torch.Size([1, target_ch[i]] + [2] * spatial_dims)

            target_params = [27818592, 49453152, 87564416, 196227264, 348143872]
            if spatial_dims == 2:
                import torchinfo
                log = torchinfo.summary(model, input_size=input_size, verbose=0)
                assert log.total_params == target_params[i]