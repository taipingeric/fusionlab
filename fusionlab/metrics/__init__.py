from fusionlab import BACKEND
if BACKEND['torch']:
    from .dicescore.dice import DiceScore
    from .iouscore.iou import IoUScore