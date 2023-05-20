import tensorflow as tf
from einops import rearrange
from fusionlab.functional.tfiou import tf_iou_score

__all__ = ["TFIoULoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"


class TFIoULoss(tf.keras.losses.Loss):
    def __init__(
        self,
        mode="binary",  # binary, multiclass
        log_loss=False,
        from_logits=False,
    ):
        """
        Implementation of IoU loss for image segmentation task.
        It supports "binary", "multiclass"
        Args:
            mode: Metric mode {'binary', 'multiclass'}
            log_loss: If True, loss computed as `-log(dice)`; otherwise `1 - dice`
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
        axis = [0]

        if self.from_logits:
            # get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            else:
                y_pred = tf.nn.sigmoid(y_pred)

        if self.mode == BINARY_MODE:
            y_true = rearrange(y_true, "... -> (...) 1")
            y_pred = rearrange(y_pred, "... -> (...) 1")
        elif self.mode == MULTICLASS_MODE:
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, num_classes)
            y_true = rearrange(y_true, "... C -> (...) C")
            y_pred = rearrange(y_pred, "... C -> (...) C")
        else:
            AssertionError("Not implemented")

        scores = tf_iou_score(y_pred, tf.cast(y_true, y_pred.dtype), axis=axis)
        scores = tf.clip_by_value(scores, clip_value_min=1e-7, clip_value_max=scores.dtype.max)
        if self.log_loss:
            loss = -tf.math.log(scores)
        else:
            loss = 1.0 - scores
        return tf.math.reduce_mean(loss)


if __name__ == '__main__':
    print("Multiclass")
    pred = tf.convert_to_tensor([[
        [1., 2., 3., 4.],
        [2., 6., 4., 4.],
        [9., 6., 3., 4.]
    ]])
    pred = rearrange(pred, "N C H -> N H C")
    true = tf.convert_to_tensor([[2, 1, 0, 2]])

    loss = TFIoULoss("multiclass", from_logits=True)(true, pred)

    print("Binary")
    pred = tf.convert_to_tensor([0.4, 0.2, 0.3, 0.5])
    pred = tf.reshape(pred, [1, 2, 2, 1])
    true = tf.convert_to_tensor([0, 1, 0, 1])
    true = tf.reshape(true, [1, 2, 2])
    loss = TFIoULoss("binary", from_logits=True)(true, pred)

    print("Binary Log loss")
    pred = tf.convert_to_tensor([0.4, 0.2, 0.3, 0.5])
    pred = tf.reshape(pred, [1, 2, 2, 1])
    true = tf.convert_to_tensor([0, 1, 0, 1])
    true = tf.reshape(true, [1, 2, 2])
    loss = TFIoULoss("binary", from_logits=True, log_loss=True)(true, pred)