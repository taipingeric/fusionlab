from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .diceloss.dice import DiceLoss, DiceCELoss
    from .iouloss.iou import IoULoss
    from .tversky.tversky import TverskyLoss
elif BACKEND == 'tf':
    from .diceloss.tfdice import TFDiceLoss, TFDiceCE
    from .iouloss.tfiou import TFIoULoss
    from .tversky.tftversky import TFTverskyLoss
else:
    print('backend not supported!!!')
