import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from fusionlab.functional import dice_score
from fusionlab.configs import EPS

__all__ = ["DiceLoss", "DiceCELoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"


class DiceCELoss(nn.Module):
    def __init__(self, w_dice=0.5, w_ce=0.5, cls_weight=None):
        """
        Dice Loss + Cross Entropy Loss
        Args:
            w_dice: weight of Dice Loss
            w_ce: weight of CrossEntropy loss
            cls_weight:
        """
        super().__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.cls_weight = cls_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=cls_weight)

    def forward(self, y_pred, y_true):
        loss_dice = self.dice(y_pred, y_true)
        loss_ce = self.ce(y_pred, y_true)
        return self.w_dice * loss_dice + self.w_ce * loss_ce


class DiceLoss(nn.Module):
    def __init__(
        self,
        mode="multiclass",  # binary, multiclass
        log_loss=False,
        from_logits=True,
    ):
        """
        Implementation of Dice loss for image segmentation task.
        It supports "binary", "multiclass"
        ref: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/dice.py
        Args:
            mode: Metric mode {'binary', 'multiclass'}
            log_loss: If True, loss computed as `-log(dice)`; otherwise `1 - dice`
            from_logits: If True assumes input is raw logits
        """
        super().__init__()
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

        scores = dice_score(y_pred, y_true.type_as(y_pred), dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(EPS))
        else:
            loss = 1.0 - scores
        return loss.mean()


if __name__ == "__main__":

    print("multiclass")
    pred = torch.tensor([[
        [1., 2., 3., 4.],
        [2., 6., 4., 4.],
        [9., 6., 3., 4.]
    ]]).view(1, 3, 4)
    true = torch.tensor([[2, 1, 0, 2]]).view(1, 4)

    dice = DiceLoss("multiclass", from_logits=True)
    loss = dice(pred, true)

    print("Binary")
    pred = torch.tensor([0.4, 0.2, 0.3, 0.5]).reshape(1, 1, 2, 2)
    true = torch.tensor([0, 1, 0, 1]).reshape(1, 2, 2)
    dice = DiceLoss("binary", from_logits=True)
    loss = dice(pred, true)

    print("Binary Logloss")
    pred = torch.tensor([0.4, 0.2, 0.3, 0.5]).reshape(1, 1, 2, 2)
    true = torch.tensor([0, 1, 0, 1]).reshape(1, 2, 2)
    dice = DiceLoss("binary", from_logits=True, log_loss=True)
    loss = dice(pred, true)

