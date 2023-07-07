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
        inputs = torch.normal(0, 1, (1, 256, 16, 16))
        layer = SEModule(256)
        outputs = layer(inputs)
        assert list(outputs.shape) == [1, 256, 16, 16]

        inputs = torch.normal(0, 1, (1, 256, 16, 16, 16))
        layer = SEModule(256, spatial_dims=3)
        outputs = layer(inputs)
        assert list(outputs.shape) == [1, 256, 16, 16, 16]