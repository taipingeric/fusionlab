import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from fusionlab.configs import EPS

__all__ = ["TverskyLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha,
        beta,
        mode="multiclass",  # binary, multiclass
        log_loss=False,
        from_logits=True,
    ):
        """
        Implementation of Tversky loss for image segmentation task.
        It supports "binary", "multiclass"
        ref: https://github.com/kornia/kornia/blob/master/kornia/losses/tversky.py
        Args:
            mode: Metric mode {'binary', 'multiclass'}
            log_loss: If True, loss computed as `-log(dice)`; otherwise `1 - dice`
            from_logits: If True assumes input is raw logits
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.from_logits = from_logits
        self.log_loss = log_loss

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (N, C, *)
        :param y_true: (N, *)
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)  # (N, C, HW)

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

        scores = tversky_score(y_pred, y_true.type_as(y_pred),
                               self.alpha, self.beta,
                               dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(EPS))
        else:
            loss = 1.0 - scores
        return loss.mean()


def tversky_score(pred, target, alpha, beta, dims):
    """
    Shape:
        - pred: :math:`(N, C, *)`
        - target: :math:`(N, C, *)`
        - Output: scalar.
    """
    assert pred.size() == target.size()

    intersection = torch.sum(pred * target, dim=dims)
    fp = torch.sum(pred * (1. - target), dims)
    fn = torch.sum((1. - pred) * target, dims)

    denominator = intersection + alpha * fp + beta * fn
    return intersection / denominator.clamp(min=EPS)


