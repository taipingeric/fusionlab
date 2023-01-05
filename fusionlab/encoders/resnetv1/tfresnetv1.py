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
    def __init__(self, cout, kernel_size=3, stride=1, activation=True, padding=True):
        super().__init__()
        self.conv = layers.Conv2D(cout, kernel_size, stride,
                                  padding='same' if padding else 'valid')
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU() if activation else Identity()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.act(x)
        return x


class Bottleneck(Model):
    def __init__(self, dims, kernel_size=3, stride=None):
        super().__init__()
        dim1, dim2, dim3 = dims
        self.conv1 = ConvBlock(dim1, kernel_size=1)
        self.conv2 = ConvBlock(dim2, kernel_size=kernel_size,
                               stride=stride if stride else 1)
        self.conv3 = ConvBlock(dim3, kernel_size=1, activation=False)
        self.act = layers.ReLU()
        self.skip = Identity() if not stride else ConvBlock(dim3,
                                                            kernel_size=1,
                                                            stride=stride,
                                                            activation=False)

    def call(self, inputs, training=None):
        identity = self.skip(inputs, training)

        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)

        x += identity
        x = self.act(x)
        return x


class TFResNet50V1(Model):
    def __init__(self):
        super(TFResNet50V1, self).__init__()
        self.conv1 = Sequential([
            ConvBlock(64, 7, stride=2),
            layers.MaxPool2D(3, strides=2, padding='same'),
        ])
        self.conv2 = Sequential([
            Bottleneck([64, 64, 256], 3, stride=1),
            Bottleneck([64, 64, 256], 3),
            Bottleneck([64, 64, 256], 3),
        ])
        self.conv3 = Sequential([
            Bottleneck([128, 128, 512], 3, stride=2),
            Bottleneck([128, 128, 512], 3),
            Bottleneck([128, 128, 512], 3),
            Bottleneck([128, 128, 512], 3),
        ])
        self.conv4 = Sequential([
            Bottleneck([256, 256, 1024], 3, stride=2),
            Bottleneck([256, 256, 1024], 3),
            Bottleneck([256, 256, 1024], 3),
            Bottleneck([256, 256, 1024], 3),
            Bottleneck([256, 256, 1024], 3),
            Bottleneck([256, 256, 1024], 3),
        ])
        self.conv5 = Sequential([
            Bottleneck([512, 512, 2048], 3, stride=2),
            Bottleneck([512, 512, 2048], 3),
            Bottleneck([512, 512, 2048], 3),
        ])

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        x = self.conv5(x, training)
        return x


if __name__ == '__main__':
    inputs = tf.random.normal((1, 224, 224, 128))
    output = Bottleneck([64, 64, 128])(inputs)
    shape = output.shape
    print("Bottleneck", shape)
    assert shape == (1, 224, 224, 128)

    output = Bottleneck([128, 128, 256], stride=1)(inputs)
    shape = output.shape
    print("Bottleneck first conv for aligh dims", shape)
    assert shape == (1, 224, 224, 256)

    output = Bottleneck([64, 64, 128], stride=2)(inputs)
    shape = output.shape
    print("Bottleneck downsample", shape)
    assert shape == (1, 112, 112, 128)

    output = Identity()(inputs)
    shape = output.shape
    print("Identity", shape)
    assert shape == (1, 224, 224, 128)


    output = TFResNet50V1()(inputs)
    shape = output.shape
    print("TFResNet50V1", shape)
    assert  shape == (1, 7, 7, 2048)




