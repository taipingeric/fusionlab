import tensorflow as tf
from einops import rearrange

__all__ = ["TFDiceLoss", "TFDiceCE"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"

# TODO: Test code
class TFDiceCE(tf.keras.losses.Loss):
    def __init__(self, mode="binary", from_logits=False, w_dice=0.5, w_ce=0.5):
        """
        Dice Loss + Cross Entropy Loss
        Args:
            w_dice: weight of Dice Loss
            w_ce: weight of CrossEntropy loss
            mode: Metric mode {'binary', 'multiclass'}
        """
        super().__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.dice = TFDiceLoss(mode, from_logits)
        if mode == BINARY_MODE:
            self.ce = tf.keras.losses.BinaryCrossentropy(from_logits)
        elif mode == MULTICLASS_MODE:
            self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits)

    def call(self, y_true, y_pred):
        loss_dice = self.dice(y_true, y_pred)
        loss_ce = self.ce(y_true, y_pred)
        return self.w_dice * loss_dice + self.w_ce * loss_ce


class TFDiceLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        mode="binary",  # binary, multiclass
        log_loss=False,
        from_logits=False,
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

    def call(self, y_true, y_pred):
        """
        :param y_true: (N, *)
        :param y_pred: (N, *, C)
        :return: scalar
        """
        y_true_shape = y_true.shape.as_list()
        y_pred_shape = y_pred.shape.as_list()
        assert y_true_shape[0] == y_pred_shape[0]
        num_classes = y_pred_shape[-1]
        axis = [0, 1]

        if self.from_logits:
            # get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            else:
                y_pred = tf.nn.sigmoid(y_pred)

        if self.mode == BINARY_MODE:
            y_true = rearrange(y_true, "N ... -> N (...) 1")
            y_pred = rearrange(y_pred, "N ... 1 -> N (...) 1")
        elif self.mode == MULTICLASS_MODE:
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, num_classes)
            y_true = rearrange(y_true, "N ... C -> N (...) C")
            y_pred = rearrange(y_pred, "N ... C -> N (...) C")
        else:
            AssertionError("Not implemented")

        scores = soft_dice_score(y_pred, tf.cast(y_true, y_pred.dtype), axis=axis)
        if self.log_loss:
            loss = -tf.math.log(tf.clip_by_value(scores, clip_value_min=1e-7, clip_value_max=scores.dtype.max))
        else:
            loss = 1.0 - scores
        return tf.math.reduce_mean(loss)


def soft_dice_score(pred, target, axis=None):
    """
    Shape:
        - Input: :math:`(*, C)` where :math:`*` means any number of additional dimensions
        - Target: :math:`(*, C)`, same shape as the input
        - Output: scalar.
    """
    eps = 1e-7
    intersection = tf.reduce_sum(pred * target, axis=axis)
    cardinality = tf.reduce_sum(pred + target, axis=axis)
    dice_score = (2.0 * intersection) / tf.clip_by_value(cardinality,
                                                         clip_value_min=eps,
                                                         clip_value_max=cardinality.dtype.max)
    return dice_score


if __name__ == '__main__':
    print("Multiclass")
    pred = tf.convert_to_tensor([[
        [1., 2., 3., 4.],
        [2., 6., 4., 4.],
        [9., 6., 3., 4.]
    ]])
    pred = rearrange(pred, "N C H -> N H C")
    true = tf.convert_to_tensor([[2, 1, 0, 2]])

    dice = TFDiceLoss("multiclass", from_logits=True)
    loss = dice(true, pred)
    assert float(loss) == 0.5519775748252869
    print(round(float(loss), 7) , '== 0.5519775748252869')

    print("Binary")
    pred = tf.convert_to_tensor([0.4, 0.2, 0.3, 0.5])
    pred = tf.reshape(pred, [1, 2, 2, 1])
    true = tf.convert_to_tensor([0, 1, 0, 1])
    true = tf.reshape(true, [1, 2, 2])
    dice = TFDiceLoss("binary", from_logits=True)
    loss = dice(true, pred)
    print(round(float(loss), 7), '== 0.46044689416885376')

    print("Binary Log loss")
    pred = tf.convert_to_tensor([0.4, 0.2, 0.3, 0.5])
    pred = tf.reshape(pred, [1, 2, 2, 1])
    true = tf.convert_to_tensor([0, 1, 0, 1])
    true = tf.reshape(true, [1, 2, 2])
    dice = TFDiceLoss("binary", from_logits=True, log_loss=True)
    loss = dice(true, pred)
    print(round(float(loss), 7), '== 0.6170140504837036')