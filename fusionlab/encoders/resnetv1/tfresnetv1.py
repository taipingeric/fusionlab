import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers

# ResNet50
# Ref:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
# https://github.com/raghakot/keras-resnet/blob/master/README.md


class Identity(layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, inputs, training=None):
        return inputs


class ConvBlock(Model):
    def __init__(self, cout, kernel_size=3, stride=1, activation=True):
        super().__init__()
        self.conv = layers.Conv2D(cout, kernel_size, stride, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU() if activation is True else Identity()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.act(x)
        return x


class Downsample(Model):
    def __init__(self, cout):
        super().__init__()
        self.conv = ConvBlock(cout, kernel_size=1, stride=2, activation=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training)
        return x


class ResidualBlock(Model):
    def __init__(self, dims, kernel_size=3, downsample=False):
        super().__init__()
        dim1, dim2, dim3 = dims
        self.conv1 = ConvBlock(dim1, kernel_size=1)
        self.conv2 = ConvBlock(dim2, kernel_size=kernel_size,
                               stride=1 if not downsample else 2)
        self.conv3 = ConvBlock(dim3, kernel_size=1, activation=False)
        self.act = layers.ReLU()
        self.skip = Identity() if not downsample else Downsample(dim3)

    def call(self, inputs, training=None):
        identity = self.skip(inputs, training)

        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)

        x += identity
        x = self.act(x)
        return x




if __name__ == '__main__':
    inputs = tf.random.normal((1, 224, 224, 128))
    output = ResidualBlock([64, 64, 128])(inputs)
    shape = output.shape
    print("ResidualBlock", shape)
    assert shape == (1, 224, 224, 128)

    output = ResidualBlock([64, 64, 128], downsample=True)(inputs)
    shape = output.shape
    print("ResidualBlock downsample", shape)
    assert shape == (1, 112, 112, 128)

    output = Identity()(inputs)
    shape = output.shape
    print("Identity", shape)
    assert shape == (1, 224, 224, 128)

    output = Downsample(256)(inputs)
    shape = output.shape
    print("Downsample", shape)
    assert shape == (1, 112, 112, 256)

    


