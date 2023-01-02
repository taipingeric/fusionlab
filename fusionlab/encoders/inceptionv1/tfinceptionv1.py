import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

# ref: https://arxiv.org/abs/1409.4842
# Going Deeper with Convolutions


class ConvBlock(Model):
    def __init__(self, cout, kernel_size=3, stride=1):
        super().__init__()
        self.conv = layers.Conv2D(cout, kernel_size, stride, padding="same")
        self.act = layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        return x


class InceptionBlock(Model):
    def __init__(self, dim0, dim1, dim2, dim3):
        super().__init__()
        self.branch1 = ConvBlock(dim0, 1)
        self.branch3 = Sequential([
            ConvBlock(dim1[0], 1),
            ConvBlock(dim1[1], 3)
        ])
        self.branch5 = Sequential([
            ConvBlock(dim2[0], 1),
            ConvBlock(dim2[1], 5)]
        )
        self.pool = Sequential([
            layers.MaxPool2D(3, 1, padding='same'),
            ConvBlock(dim3, 1)
        ])

    def call(self, inputs):
        x = inputs
        x0 = self.branch1(x)
        x1 = self.branch3(x)
        x2 = self.branch5(x)
        x3 = self.pool(x)
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        return x


class TFInceptionNetV1(Model):
    def __init__(self):
        super().__init__()
        self.stem = Sequential([
            ConvBlock(64, 7, stride=2),
            layers.MaxPool2D(3, 2, padding='same'),
            ConvBlock(192, 3),
            layers.MaxPool2D(3, 2, padding='same'),
        ])
        self.incept3a = InceptionBlock(64, (96, 128), (16, 32), 32)
        self.incept3b = InceptionBlock(128, (128, 192), (32, 96), 64)
        self.pool3 = layers.MaxPool2D(3, 2, padding='same')
        self.incept4a = InceptionBlock(192, (96, 208), (16, 48), 64)
        self.incept4b = InceptionBlock(160, (112, 224), (24, 64), 64)
        self.incept4c = InceptionBlock(128, (128, 256), (24, 64), 64)
        self.incept4d = InceptionBlock(112, (144, 288), (32, 64), 64)
        self.incept4e = InceptionBlock(256, (160, 320), (32, 128), 128)
        self.pool4 = layers.MaxPool2D(3, 2, padding='same')
        self.incept5a = InceptionBlock(256, (160, 320), (32, 128), 128)
        self.incept5b = InceptionBlock(384, (192, 384), (48, 128), 128)

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.incept3a(x)
        x = self.incept3b(x)
        x = self.pool3(x)
        x = self.incept4a(x)
        x = self.incept4b(x)
        x = self.incept4c(x)
        x = self.incept4d(x)
        x = self.incept4e(x)
        x = self.pool4(x)
        x = self.incept5a(x)
        x = self.incept5b(x)
        return x


if __name__ == "__main__":
    inputs = tf.random.normal((1, 224, 224, 3))
    output = InceptionBlock(64, (96, 128), (16, 32), 32)(inputs)
    shape = output.shape
    print("InceptionBlock", shape)
    assert shape == (1, 224, 224, 256)

    output = TFInceptionNetV1()(inputs)
    shape = output.shape
    print("TFInceptionNetV1", shape)
    assert shape == (1, 7, 7, 1024)
