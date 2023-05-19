from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .dcgan import *
    from .trainer import *
elif BACKEND == 'tf':
    print('TF trainer is under construction')
else:
    print('backend not supported!!!')