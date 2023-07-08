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
        from fusionlab.encoders import (
            EfficientNetB0,
            EfficientNetB1,
            EfficientNetB2,
            EfficientNetB3,
            EfficientNetB4,
            EfficientNetB5,
            EfficientNetB6,
            EfficientNetB7,
        )
        import torchinfo

        cin = 3
        for dim in [1, 2, 3]:
            for i in range(0, 8):
                eff = eval(f'EfficientNetB{i}')
                model = eff(dim, cin=cin)
                img_size = 64
                size = tuple([1, cin] + [img_size] * dim)
                inputs = torch.randn(size)
                outputs = model(inputs)
                target_ch = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
                target_size = [img_size // 32]*dim
                assert outputs.shape == (1, target_ch[i], *target_size)

                if dim == 2:
                    target_params = [4007548, 6513184, 7700994, 10696232, 17548616, 28340784, 40735704, 63786960]
                    log = torchinfo.summary(model, (1, cin, img_size, img_size))
                    assert log.total_params == target_params[i]
    
    def test_convnext(self):
        from fusionlab.encoders import (
            ConvNeXtTiny,
            ConvNeXtSmall,
            ConvNeXtBase,
            ConvNeXtLarge,
            ConvNeXtXLarge
        )
        import torchinfo
        for spatial_dims in [1, 2, 3]:
            for i, convnext_type in enumerate(['Tiny', 'Small', 'Base', 'Large', 'XLarge']):
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
