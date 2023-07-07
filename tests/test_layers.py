from fusionlab.layers import ConvNormAct

class TestConvNormAct:
    def test_ConvNormAct(self):
        import torch
        inputs = torch.randn(1, 3, 16)
        l = ConvNormAct(1, 3, 4, 3, 1, 1)
        outputs = l(inputs)
        assert outputs.shape == torch.Size([1, 4, 16])

        inputs = torch.randn(1, 3, 16, 16)
        l = ConvNormAct(2, 3, 4, 3, 1, 1)
        outputs = l(inputs)
        assert outputs.shape == torch.Size([1, 4, 16, 16])

        inputs = torch.randn(1, 3, 16, 16, 16)
        l = ConvNormAct(3, 3, 4, 3, 1, 1)
        outputs = l(inputs)
        assert outputs.shape == torch.Size([1, 4, 16, 16, 16])

class TestSqueezeExcitation:
    def test_se(self):
        import torch
        from fusionlab.layers import SEModule
        cin = 64
        squeeze_channels = 16

        for i in range(1, 4):
            size = tuple([1, cin] + [16] * i)
            inputs = torch.randn(size)
            layer = SEModule(cin, squeeze_channels, spatial_dims=i)
            outputs = layer(inputs)
            assert outputs.shape == size