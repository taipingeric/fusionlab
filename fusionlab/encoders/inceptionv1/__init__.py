from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .inceptionv1 import InceptionNetV1
elif BACKEND == 'tf':
    from .tfinceptionv1 import TFInceptionNetV1
else:
    print('backend not supported!!!')