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