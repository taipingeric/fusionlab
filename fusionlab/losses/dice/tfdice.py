import tensorflow as tf

__all__ = ["TFDiceLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"

# class TFDiceCE(tf.keras.losses.Loss):
#     def __init__(self, cls_weight):
#         self.cls_weight = cls_weight
#         self.dice = DiceLoss()
#         self.ce = nn.CrossEntropyLoss(weight=cls_weight)

#     def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
#         loss_dice = self.dice(y_pred, y_true)
#         loss_ce = self.ce(y_pred, y_true)
#         return 0.5*loss_dice + 0.5*loss_ce


class TFDiceLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        mode="binary",  # binary, multiclass
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

    def call(self, y_pred, y_true):
        """
        :param y_pred: (N, H, W, C)
        :param y_true: (N, H, W)
        :return: scalar
        """
        # assert y_true.size(0) == y_pred.size(0)
        y_true_shape = y_true.shape.as_list()
        y_pred_shape = y_pred.shape.as_list()
        assert y_true_shape[0] == y_pred_shape[0]
        bs = y_true_shape[0]
        num_classes = y_pred_shape[-1]
        dims = [0, 1]  # (N, H*W)

        if self.from_logits:
            # get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            else:
                y_pred = tf.nn.sigmoid(y_pred)

        if self.mode == BINARY_MODE:
            y_true = tf.reshape(y_true, [bs, -1, 1])
            y_pred = tf.reshape(y_pred, [bs, -1, 1])
        elif self.mode == MULTICLASS_MODE:
            y_true = tf.reshape(y_true, [bs, -1])
            y_true = tf.one_hot(y_true, num_classes)  # N, H*W -> N, H*W, C
            y_pred = tf.reshape(y_pred, [bs, -1, num_classes])
        else:
            AssertionError("Not implemented")

        scores = soft_dice_score(y_pred, tf.cast(y_true, y_pred.dtype), dims=dims)
        if self.log_loss:
            loss = -tf.math.log(tf.clip_by_value(scores, clip_value_min=1e-7))
        else:
            loss = 1.0 - scores
        return tf.math.reduce_mean(loss)


def soft_dice_score(pred, target, dims=None):
    """
    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means any number of additional dimensions
        - Target: :math:`(N, C, *)`, same shape as the input
        - Output: scalar.
    """
    pred_shape = pred.shape.as_list()
    target_shape = target.shape.as_list()
    assert pred_shape == target_shape
    eps = 1e-7
    intersection = tf.reduce_sum(pred * target, axis=dims)
    cardinality = tf.reduce_sum(pred + target, axis=dims)
    dice_score = (2.0 * intersection) / tf.clip_by_value(cardinality, clip_value_min=eps, clip_value_max=cardinality.dtype.max)
    return dice_score


if __name__ == '__main__':
    pred = tf.convert_to_tensor([[
        [1., 2., 3.],
        [2., 6., 4.],
        [9., 6., 3.],
        [4., 8., 12.]
    ]])
    true = tf.convert_to_tensor([[2, 1, 0, 2]])
    dice = TFDiceLoss("multiclass", from_logits=True)
    loss = dice(pred, true)
    print(loss, "== 0.13497286?")

    print("Binary")
    pred = tf.convert_to_tensor([0.4, 0.2, 0.3, 0.5])
    pred = tf.reshape(pred, [1, 2, 2, 1])
    true = tf.convert_to_tensor([0, 1, 0, 1])
    true = tf.reshape(true, [1, 2, 2])

    print(pred.shape, true.shape)
    dice = TFDiceLoss("binary", from_logits=True)
    loss = dice(pred, true)
    print(loss, '== 0.4604469')