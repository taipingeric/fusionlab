from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .squeeze_excitation.se import SEModule
elif BACKEND == 'tf':
    from .squeeze_excitation.tfse import TFSEModule
else:
    print('backend not supported!!!')