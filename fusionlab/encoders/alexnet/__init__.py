from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .alexnet import AlexNet
elif BACKEND == 'tf':
    from .tfalexnet import TFAlexNet
else:
    print('backend not supported!!!')
