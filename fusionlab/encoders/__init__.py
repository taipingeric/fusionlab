from fusionlab import BACKEND
if BACKEND['torch']:
    from .alexnet.alexnet import AlexNet
    from .vgg.vgg import VGG16, VGG19
    from .inceptionv1.inceptionv1 import InceptionNetV1
    from .resnetv1.resnetv1 import *
    from .efficientnet.efficientnet import (
        EfficientNet,
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
        ConvNeXt,
        ConvNeXtTiny,
        ConvNeXtSmall,
        ConvNeXtBase,
        ConvNeXtLarge,
        ConvNeXtXLarge
    )
if BACKEND['tf']:
    from .alexnet.tfalexnet import TFAlexNet
    from .vgg.tfvgg import TFVGG16, TFVGG19
    from .inceptionv1.tfinceptionv1 import TFInceptionNetV1
    from .resnetv1.tfresnetv1 import TFResNet50V1
