from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .factories import *
    from .squeeze_excitation.se import SEModule
    from .base import ConvNormAct
elif BACKEND == 'tf':
    from .squeeze_excitation.tfse import TFSEModule
else:
    print('backend not supported!!!')