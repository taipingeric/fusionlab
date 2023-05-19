from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .alexnet.alexnet import AlexNet
    from .vgg.vgg import VGG16, VGG19
    from .inceptionv1.inceptionv1 import InceptionNetV1
    from .resnetv1.resnetv1 import ResNet50V1
elif BACKEND == 'tf':
    from .alexnet.tfalexnet import TFAlexNet
    from .vgg.tfvgg import TFVGG16, TFVGG19
    from .inceptionv1.tfinceptionv1 import TFInceptionNetV1
    from .resnetv1.tfresnetv1 import TFResNet50V1
else:
    print('backend not supported!!!')
