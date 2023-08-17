from fusionlab import BACKEND
if BACKEND['torch']:
    from .dicescore.dice import DiceScore, JaccardScore
    from .iouscore.iou import IoUScore