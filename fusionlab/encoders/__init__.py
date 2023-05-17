from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .alexnet import AlexNet
    from .vgg import VGG16, VGG19
    from .inceptionv1 import InceptionNetV1
    from .resnetv1 import ResNet50V1
elif BACKEND == 'tf':
    from .alexnet import TFAlexNet
    from .vgg import TFVGG16, TFVGG19
    from .inceptionv1 import TFInceptionNetV1
    from .resnetv1 import TFResNet50V1
else:
    print('backend not supported!!!')
