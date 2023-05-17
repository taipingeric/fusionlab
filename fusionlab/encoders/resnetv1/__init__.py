from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .resnetv1 import ResNet50V1
elif BACKEND == 'tf':
    from .tfresnetv1 import TFResNet50V1
else:
    print('backend not supported!!!')