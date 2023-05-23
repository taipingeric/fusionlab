from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .base import *
    from .lstm import *
    from .vgg import *
elif BACKEND == 'tf':
    print('not built yet')
else:
    print('backend not supported!!!')

    
