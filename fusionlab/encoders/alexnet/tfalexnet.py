import tensorflow as tf


class TFAlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(2),
            tf.keras.layers.Conv2D(64, kernel_size=11, strides=4),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(192, kernel_size=5, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(384, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        ])

    def call(self, inputs):
        return self.features(inputs)


if __name__ == '__main__':
    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFAlexNet()(inputs)
    shape = output.shape
    print(shape)
    assert shape[1:3] == [6, 6]
