import torch
import tensorflow as tf
from fusionlab.configs import EPS

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

tf_jaccard_score = tf_iou_score
