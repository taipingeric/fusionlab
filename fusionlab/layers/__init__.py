from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .squeeze_excitation import SEModule
elif BACKEND == 'tf':
    from .squeeze_excitation import TFSEModule
else:
    print('backend not supported!!!')