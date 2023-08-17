import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from fusionlab.functional import iou_score

__all__ = ["IoUScore"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"

class IoUScore(nn.Module):
    def __init__(
        self,
        mode="multiclass",  # binary, multiclass
        from_logits=True,
        reduction="none", # mean, none
    ):
        """
        Implementation of Iou score for segmentation task.
        It supports "binary", "multiclass"
        Args:
            mode: Metric mode {'binary', 'multiclass'}
            from_logits: If True assumes input is raw logits
            reduction: "mean" or "none", if "none" returns dice score for each channels, else returns mean
        """
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (N, C, *)
        :param y_true: (N, *)
        :return: (C, ) if mode is 'multiclass' else (1, )
        """
        assert y_true.size(0) == y_pred.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)  # (N, C, *)

        if self.from_logits:
            # get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = F.softmax(y_pred, dim=1)
            else:
                y_pred = torch.sigmoid(y_pred)

        if self.mode == BINARY_MODE:
            y_true = rearrange(y_true, "N ... -> N 1 (...)")
            y_pred = rearrange(y_pred, "N 1 ... -> N 1 (...)")
        elif self.mode == MULTICLASS_MODE:
            y_pred = rearrange(y_pred, "N C ... -> N C (...)")
            y_true = F.one_hot(y_true, num_classes)  # (N, *) -> (N, *, C)
            y_true = rearrange(y_true, "N ... C -> N C (...)")
        else:
            AssertionError("Not implemented")

        scores = iou_score(y_pred, y_true.type_as(y_pred), dims=dims)
        if self.reduction == "none":
            return scores
        else:
            return scores.mean()