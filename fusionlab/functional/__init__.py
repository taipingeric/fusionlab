from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .dice import dice_score
    from .iou import (
        iou_score,
        jaccard_score,
    )
elif BACKEND == 'tf':
    from .tfdice import tf_dice_score
    from .tfiou import (
        tf_iou_score,
        tf_jaccard_score,
    )
else:
    print('backend not supported!!!')