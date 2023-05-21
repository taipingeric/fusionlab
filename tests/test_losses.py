import torch
import tensorflow as tf
from einops import rearrange
from fusionlab.losses import *
from pytest import approx

EPS = 1e-6


class Data:
    def __init__(self):
        self.pred = [[
            [1., 2., 3., 4.],
            [2., 6., 4., 4.],
            [9., 6., 3., 4.]
        ]]
        self.target = [[2, 1, 0, 2]]


class BinaryData:
    def __init__(self):
        self.pred = [0.4, 0.2, 0.3, 0.5]
        self.target = [0, 1, 0, 1]


class TestSegLoss:
    def test_dice_loss(self):
        from fusionlab.losses.diceloss.tfdice import TFDiceLoss
        # Multi class
        data = Data()
        true_loss = 0.5519775748252869
        # PT
        pred = torch.tensor(data.pred).view(1, 3, 4)
        true = torch.tensor(data.target).view(1, 4)
        loss = DiceLoss("multiclass", from_logits=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = rearrange(pred, "N C H -> N H C")
        true = tf.convert_to_tensor(data.target)
        loss = TFDiceLoss("multiclass", from_logits=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)

        # Binary Loss
        data = BinaryData()
        true_loss = 0.46044695377349854

        pred = torch.tensor(data.pred).reshape(1, 1, 2, 2)
        true = torch.tensor(data.target).reshape(1, 2, 2)
        # PT
        loss = DiceLoss("binary", from_logits=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = tf.reshape(pred, [1, 2, 2, 1])
        true = tf.convert_to_tensor(data.target)
        true = tf.reshape(true, [1, 2, 2])
        loss = TFDiceLoss("binary", from_logits=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)

        # Binary Log loss
        true_loss = 0.6170141696929932

        # PT
        pred = torch.tensor(data.pred).reshape(1, 1, 2, 2)
        true = torch.tensor(data.target).reshape(1, 2, 2)
        loss = DiceLoss("binary", from_logits=True, log_loss=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = tf.reshape(pred, [1, 2, 2, 1])
        true = tf.convert_to_tensor(data.target)
        true = tf.reshape(true, [1, 2, 2])
        loss = TFDiceLoss("binary", from_logits=True, log_loss=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)

    def test_iou_loss(self):
        from fusionlab.losses.iouloss.tfiou import TFIoULoss
        # multi class
        true_loss = 0.6969285607337952
        data = Data()

        # PT
        pred = torch.tensor(data.pred).view(1, 3, 4)
        true = torch.tensor(data.target).view(1, 4)
        loss = IoULoss("multiclass", from_logits=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = rearrange(pred, "N C H -> N H C")
        true = tf.convert_to_tensor(data.target)
        loss = TFIoULoss("multiclass", from_logits=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)

        # Binary
        data = BinaryData()
        true_loss = 0.6305561661720276
        # PT
        pred = torch.tensor(data.pred).reshape(1, 1, 2, 2)
        true = torch.tensor(data.target).reshape(1, 2, 2)
        loss = IoULoss("binary", from_logits=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = tf.reshape(pred, [1, 2, 2, 1])
        true = tf.convert_to_tensor(data.target)
        true = tf.reshape(true, [1, 2, 2])
        loss = TFIoULoss("binary", from_logits=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)

        # Binary Log loss
        data = BinaryData()
        true_loss = 0.9957565665245056
        # PT
        pred = torch.tensor(data.pred).reshape(1, 1, 2, 2)
        true = torch.tensor(data.target).reshape(1, 2, 2)
        loss = IoULoss("binary", from_logits=True, log_loss=True)(pred, true)
        assert loss == approx(true_loss, EPS)
        # TF
        pred = tf.convert_to_tensor(data.pred)
        pred = tf.reshape(pred, [1, 2, 2, 1])
        true = tf.convert_to_tensor(data.target)
        true = tf.reshape(true, [1, 2, 2])
        loss = TFIoULoss("binary", from_logits=True, log_loss=True)(true, pred)
        assert float(loss) == approx(true_loss, EPS)