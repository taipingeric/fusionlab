import fusionlab
fusionlab.configs.BACKEND = 'torch'
import torch
from fusionlab.encoders import AlexNet, VGG16, VGG19, InceptionNetV1, ResNet50V1

class TestEncoders:
    def test_AlexNet(self):
        print("Test AlexNet")
        model = fusionlab.encoders.AlexNet()
        assert model is not None

        img_size = 224
        # 2D
        inputs = torch.rand(1, 3, img_size, img_size)
        output = AlexNet(spatial_dims=2, cin=3)(inputs)
        assert list(output.shape) == [1, 256, 6, 6]
        # 1D
        inputs = torch.rand(1, 3, img_size)
        output = AlexNet(spatial_dims=1, cin=3)(inputs)
        assert list(output.shape) == [1, 256, 6]
        # 3D

        img_size = 128
        inputs = torch.normal(0, 1, (1, 3, img_size, img_size, img_size))
        output = AlexNet(spatial_dims=3, cin=3)(inputs)
        assert list(output.shape) == [1, 256, 3, 3, 3]

    def test_VGG(self):
        size = 224
        print("Test VGG")
        # 1D
        # VGG16
        inputs = torch.rand(1, 3, size)
        output = VGG16(spatial_dims=1, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7]

        # VGG19
        inputs = torch.rand(1, 3, size)
        output = VGG19(spatial_dims=1, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7]

        # 2D
        # VGG16
        inputs = torch.rand(1, 3, size, size)
        output = VGG16(spatial_dims=2, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7, 7]

        # VGG19
        inputs = torch.rand(1, 3, size, size)
        output = VGG19(spatial_dims=2, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7, 7]

        #3D
        size = 64
        # VGG16
        inputs = torch.rand(1, 3, size, size, size)
        output = VGG16(spatial_dims=3, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [2, 2, 2]

        # VGG19
        inputs = torch.rand(1, 3, size, size, size)
        output = VGG19(spatial_dims=3, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [2, 2, 2]

    def test_InceptionNetV1(self):
        
        print("Test InceptionNetV1")
        size = 224
        # 1D
        inputs = torch.rand(1, 3, size)
        output = InceptionNetV1(spatial_dims=1, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7]

        # 2D
        inputs = torch.rand(1, 3, size, size)
        output = InceptionNetV1(spatial_dims=2, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7, 7]

        # 3D
        size = 64
        inputs = torch.rand(1, 3, size, size, size)
        output = InceptionNetV1(spatial_dims=3, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [2, 2, 2]
    
    def test_ResNet50V1(self):
        print("Test ResNet50V1")
        size = 224
        # 1D
        inputs = torch.rand(1, 3, size)
        output = ResNet50V1(spatial_dims=1, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7]

        # 2D
        inputs = torch.rand(1, 3, size, size)
        output = ResNet50V1(spatial_dims=2, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [7, 7]

        # 3D
        size = 64
        inputs = torch.rand(1, 3, size, size, size)
        output = ResNet50V1(spatial_dims=3, cin=3)(inputs)
        shape = list(output.shape)
        assert shape[2:] == [2, 2, 2]

    def test_efficientnet(self):
        from fusionlab.encoders import EfficientNetB0

        cin = 3
        for i in range(1, 4):
            print(f"Test EfficientNetB0 with {i}D")
            size = tuple([1, cin] + [64] * i)
            inputs = torch.randn(size)
            target_size = tuple([1, 1280] + [2] * i)
            model = EfficientNetB0(spatial_dims=i, cin=cin)
            outputs = model(inputs)
            assert outputs.shape == target_size
