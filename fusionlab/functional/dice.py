from typing import Tuple
import torch
from fusionlab.configs import EPS


def dice_score(pred: torch.Tensor, 
               target: torch.Tensor, 
               dims: Tuple[int, ...]=None) -> torch.Tensor:
    """
    Computes the dice score

    Args:
        pred: (N, C, *)
        target: (N, C, *)
        dims: dimensions to sum over

    """
    assert pred.size() == target.size()
    intersection = torch.sum(pred * target, dim=dims)
    cardinality = torch.sum(pred + target, dim=dims)
    return (2.0 * intersection) / cardinality.clamp(min=EPS)