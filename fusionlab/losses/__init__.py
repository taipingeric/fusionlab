from fusionlab import BACKEND
if BACKEND['torch']:
    from .diceloss.dice import DiceLoss, DiceCELoss
    from .iouloss.iou import IoULoss
    from .tversky.tversky import TverskyLoss
if BACKEND['tf']:
    from .diceloss.tfdice import TFDiceLoss, TFDiceCE
    from .iouloss.tfiou import TFIoULoss
    from .tversky.tftversky import TFTverskyLoss