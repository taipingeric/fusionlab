import torch

def generate_inputs(img_size, spatial_dims):
    size = tuple([1, 3] + [img_size] * spatial_dims)
    return torch.randn(size)

class TestSeg:
    def test_unet(self):
        from fusionlab.segmentation import UNet
        for i in range(1, 4):
            inputs = generate_inputs(64, i)
            model = UNet(3, 2, spatial_dims=i)
            outputs = model(inputs)
            assert outputs.shape == tuple([1, 2] + [64] * i)
    
    def test_resunet(self):
        from fusionlab.segmentation import ResUNet
        for i in range(1, 4):
            inputs = generate_inputs(64, i)
            model = ResUNet(3, 2, spatial_dims=i)
            outputs = model(inputs)
            assert outputs.shape == tuple([1, 2] + [64] * i)
    
    def test_unet2plus(self):
        from fusionlab.segmentation import UNet2plus
        for i in range(1, 4):
            inputs = generate_inputs(64, i)
            model = UNet2plus(3, 2, 16, spatial_dims=i)
            outputs = model(inputs)
            assert outputs.shape == tuple([1, 2] + [64] * i)
    
    def test_transunet(self):
        from fusionlab.segmentation import TransUNet
        for i in [2]:
            inputs = generate_inputs(64, i)
            model = TransUNet(
                in_channels=3, 
                num_classes=2, 
                spatial_dims=i
            )
            outputs = model(inputs)
            assert outputs.shape == tuple([1, 2] + [64] * i)
    
    def test_unetr(self):
        from fusionlab.segmentation import UNETR
        for i in range(1, 4):
            inputs = generate_inputs(64, i)
            model = UNETR(3, 2, 64, spatial_dims=i)
            outputs = model(inputs)
            assert outputs.shape == tuple([1, 2] + [64] * i)

    def test_segformer(self):
        from fusionlab.segmentation import SegFormer
        inputs = torch.rand(1, 3, 64, 64)
        for i in range(6):
            mit_type = f'B{i}'
            model = SegFormer(num_classes=2, mit_encoder_type=mit_type)
            outputs = model(inputs)
            assert outputs.shape == (1, 2, 64, 64)