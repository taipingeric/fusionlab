from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .unet.unet import UNet
    from .resunet.resunet import ResUNet
    from .unet2plus.unet2plus import UNet2plus
    from .base import HFSegmentationModel
elif BACKEND == 'tf':
    from .unet.tfunet import TFUNet
    from .resunet.tfresunet import TFResUNet
    from .unet2plus.tfunet2plus import TFUNet2plus
else:
    print('backend not supported!!!')