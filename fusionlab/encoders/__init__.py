from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .alexnet.alexnet import AlexNet
    from .vgg.vgg import VGG16, VGG19
    from .inceptionv1.inceptionv1 import InceptionNetV1
    from .resnetv1.resnetv1 import ResNet50V1
    from .efficientnet.efficientnet import (
        EfficientNetB0, 
        EfficientNetB1,
        EfficientNetB2,
        EfficientNetB3,
        EfficientNetB4,
        EfficientNetB5,
        EfficientNetB6,
        EfficientNetB7
    )
    from .convnext.convnext import (
        ConvNeXtTiny,
        ConvNeXtSmall,
        ConvNeXtBase,
        ConvNeXtLarge,
        ConvNeXtXLarge
    )
elif BACKEND == 'tf':
    from .alexnet.tfalexnet import TFAlexNet
    from .vgg.tfvgg import TFVGG16, TFVGG19
    from .inceptionv1.tfinceptionv1 import TFInceptionNetV1
    from .resnetv1.tfresnetv1 import TFResNet50V1
else:
    print('backend not supported!!!')
