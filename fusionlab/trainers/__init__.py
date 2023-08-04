from fusionlab import BACKEND
if BACKEND['torch']:
    from .dcgan import *
    from .trainer import *