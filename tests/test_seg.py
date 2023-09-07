import torch

class TestSeg:
    def test_segformer(self):
        from fusionlab.segmentation import SegFormer
        inputs = torch.rand(1, 3, 128, 128)
        for i in range(6):
            mit_type = f'B{i}'
            model = SegFormer(num_classes=2, mit_encoder_type=mit_type)
            outputs = model(inputs)
            assert outputs.shape == (1, 2, 128, 128)