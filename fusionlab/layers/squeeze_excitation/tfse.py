import tensorflow as tf
from tensorflow.keras import layers, Sequential


class TFSEModule(layers.Layer):
    def __init__(self, cin, ratio=16):
        super().__init__()
        cout = int(cin / ratio)
        self.gate = Sequential([
            layers.Conv2D(cout, kernel_size=1),
            layers.ReLU(),
            layers.Conv2D(cin, kernel_size=1),
            layers.Activation(tf.nn.sigmoid),
        ])

    def call(self, inputs):
        x = tf.reduce_mean(inputs, (1, 2), keepdims=True)
        x = self.gate(x)
        return inputs * x


if __name__ == '__main__':
    inputs = tf.random.normal((1, 224, 224, 256), 0, 1)
    layer = TFSEModule(256)
    outputs = layer(inputs)
    assert list(outputs.shape) == [1, 224, 224, 256]
