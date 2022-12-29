import tensorflow as tf

# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
class TFVGG16(tf.keras.Model):
    def __init__(self):
        super().__init__()
        ksize = 3
        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(128, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
        ])

    def call(self, inputs):
        return self.features(inputs)


class TFVGG19(tf.keras.Model):
    def __init__(self):
        super().__init__()
        ksize = 3
        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(128, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, ksize, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
        ])

    def call(self, inputs):
        return self.features(inputs)


if __name__ == '__main__':
    # VGG16
    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFVGG16()(inputs)
    shape = output.shape
    assert shape[1:3] == [7, 7]

    # VGG19
    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFVGG19()(inputs)
    shape = output.shape
    assert shape[1:3] == [7, 7]