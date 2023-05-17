import torch
from fusionlab.configs import EPS


def iou_score(pred, target, dims=None):
    """
    Shape:
        - pred: :math:`(N, C, *)`
        - target: :math:`(N, C, *)`
        - Output: scalar.
    """
    assert pred.size() == target.size()

    intersection = torch.sum(pred * target, dim=dims)
    cardinality = torch.sum(pred + target, dim=dims)
    union = cardinality - intersection
    iou = intersection / union.clamp_min(EPS)
    return iou

jaccard_score = iou_score