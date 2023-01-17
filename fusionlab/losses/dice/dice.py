import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["DiceLoss", "DiceCE"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"


class DiceCE(nn.Module):
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
        https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/dice.py
        Args:
            mode: Metric mode {'binary', 'multiclass'}
            log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
            from_logits: If True assumes input is raw logits
        """
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.log_loss = log_loss

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (N, C, H, W)
        :param y_true: (N, H, W)
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)  # (N, -1, H*W)

        if self.from_logits:
            # get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = F.softmax(y_pred, dim=1)
            else:
                y_pred = torch.sigmoid(y_pred)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.view(bs, num_classes, -1) # N, C, HW

            y_true = y_true.view(bs, -1)
            y_true = F.one_hot(y_true, num_classes)  # N, HW -> N, HW, C
            y_true = y_true.permute(0, 2, 1)  # N, C, HW
        else:
            AssertionError("Not implemented")
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(1e-7))
        else:
            loss = 1.0 - scores
        return loss.mean()


def soft_dice_score(pred, target, dims=None):
    """
    Shape:
        - Input: :math:`(N, C, *)`
        - Target: :math:`(N, C, *)`
        - Output: scalar.
    """
    assert pred.size() == target.size()
    eps = 1e-7
    intersection = torch.sum(pred * target, dim=dims)
    cardinality = torch.sum(pred + target, dim=dims)
    dice_score = (2.0 * intersection) / cardinality.clamp(min=eps)
    return dice_score


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
    assert loss.item() == 0.5519775748252869
    print(loss.item()) # 0.5519775748252869

    print("Binary")
    pred = torch.tensor([0.4, 0.2, 0.3, 0.5]).reshape(1, 1, 2, 2)
    true = torch.tensor([0, 1, 0, 1]).reshape(1, 2, 2)
    dice = DiceLoss("binary", from_logits=True)
    loss = dice(pred, true)
    print(loss.item()) #0.46044689416885376

    print("Binary Logloss")
    pred = torch.tensor([0.4, 0.2, 0.3, 0.5]).reshape(1, 1, 2, 2)
    true = torch.tensor([0, 1, 0, 1]).reshape(1, 2, 2)
    dice = DiceLoss("binary", from_logits=True, log_loss=True)
    loss = dice(pred, true)
    print(loss.item()) #0.6170140504837036

