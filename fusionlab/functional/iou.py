import torch
import tensorflow as tf
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


def tf_iou_score(pred, target, axis=None):
    """
    Shape:
        - pred: :math:`(N, *, C)` where :math:`*` means any number of additional dimensions
        - target: :math:`(N, *, C)`, same shape as the input
        - Output: scalar.
    """
    intersection = tf.reduce_sum(pred * target, axis=axis)
    cardinality = tf.reduce_sum(pred + target, axis=axis)
    cardinality = tf.clip_by_value(cardinality,
                                   clip_value_min=EPS,
                                   clip_value_max=cardinality.dtype.max)
    union = cardinality - intersection
    return intersection / union


jaccard_score = iou_score
tf_jaccard_score = tf_iou_score
