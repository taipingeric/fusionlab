from . import (
    functional,
    encoders,
    utils,
    datasets,
    layers,
    classification,
    segmentation,
    losses,
    trainers,
    configs
)
from .__version__ import __version__


def set_backend(backend):
    assert backend in ('torch', 'tf'), 'Error: the backend should be either torch or tf'
    import importlib.util
    if backend == 'torch':
        assert importlib.util.find_spec("torch") is not None, 'Error: pytorch is not installed'
    elif backend == 'tf':
        assert importlib.util.find_spec("tensorflow") is not None, 'Error: tensorflow is not installed'

    cfg_dict = {c: configs.__getattribute__(c) for c in configs.__dir__() if '__' not in c}
    cfg_dict['BACKEND'] = backend
    with open(".configs.py", "w") as f:
        for k, v in cfg_dict.items():
            if type(v) == str:
                f.write(f"{k} = \'{v}\' \n")
            else:
                f.write(f"{k} = {v} \n")

    print(f"Setting backend to {backend}, please reaload")