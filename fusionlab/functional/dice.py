import torch
import tensorflow as tf


def dice_score(pred, target, dims=None):
    """
    Shape:
        - pred: :math:`(N, C, *)`
        - target: :math:`(N, C, *)`
        - Output: scalar.
    """
    assert pred.size() == target.size()
    eps = 1e-7
    intersection = torch.sum(pred * target, dim=dims)
    cardinality = torch.sum(pred + target, dim=dims)
    dice_score = (2.0 * intersection) / cardinality.clamp(min=eps)
    return dice_score


def tf_dice_score(pred, target, axis=None):
    """
    Shape:
        - pred: :math:`(N, *, C)` where :math:`*` means any number of additional dimensions
        - target: :math:`(N, *, C)`, same shape as the input
        - Output: scalar.
    """
    eps = 1e-7
    intersection = tf.reduce_sum(pred * target, axis=axis)
    cardinality = tf.reduce_sum(pred + target, axis=axis)
    dice_score = (2.0 * intersection) / tf.clip_by_value(cardinality,
                                                         clip_value_min=eps,
                                                         clip_value_max=cardinality.dtype.max)
    return dice_score
