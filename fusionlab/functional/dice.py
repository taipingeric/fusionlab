import torch
from fusionlab.configs import EPS


def dice_score(pred, target, dims=None):
    """
    Shape:
        - pred: :math:`(N, C, *)`
        - target: :math:`(N, C, *)`
        - Output: scalar.
    """
    assert pred.size() == target.size()
    intersection = torch.sum(pred * target, dim=dims)
    cardinality = torch.sum(pred + target, dim=dims)
    return (2.0 * intersection) / cardinality.clamp(min=EPS)