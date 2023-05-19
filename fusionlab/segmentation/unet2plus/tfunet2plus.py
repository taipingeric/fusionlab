import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from fusionlab.segmentation.tfbase import TFSegmentationModel


class TFUNet2plus(TFSegmentationModel):
    def __init__(self, num_cls, base_dim):
        super().__init__()
        self.encoder = Encoder(base_dim)
        self.bridger = Bridger()
        self.decoder = Decoder(base_dim)
        self.head = Head(num_cls)


class BasicBlock(Sequential):
    def __init__(self, cout):
        conv1 = Sequential([
            layers.Conv2D(cout, 3, 1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
        ])
        conv2 = Sequential([
            layers.Conv2D(cout, 3, 1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu),
        ])
        super().__init__([conv1, conv2])


class Encoder(Model):
    def __init__(self, base_dim):
        """
        UNet Encoder
        Args:
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        self.pool = layers.MaxPool2D()
        self.conv0_0 = BasicBlock(base_dim)
        self.conv1_0 = BasicBlock(base_dim * 2)
        self.conv2_0 = BasicBlock(base_dim * 4)
        self.conv3_0 = BasicBlock(base_dim * 8)
        self.conv4_0 = BasicBlock(base_dim * 16)

    def call(self, x, training):
        x0_0 = self.conv0_0(x, training)
        x1_0 = self.conv1_0(self.pool(x0_0), training)
        x2_0 = self.conv2_0(self.pool(x1_0), training)
        x3_0 = self.conv3_0(self.pool(x2_0), training)
        x4_0 = self.conv4_0(self.pool(x3_0), training)
        return [x0_0, x1_0, x2_0, x3_0, x4_0]


class Bridger(Model):
    def call(self, x, training=None):
        return [tf.identity(i) for i in x]


class Decoder(Model):
    def __init__(self, base_dim):
        super().__init__()
        dims = [base_dim*(2**i) for i in range(5)]  # [base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*16]
        self.conv0_1 = BasicBlock(dims[0])
        self.conv1_1 = BasicBlock(dims[1])
        self.conv2_1 = BasicBlock(dims[2])
        self.conv3_1 = BasicBlock(dims[3])

        self.conv0_2 = BasicBlock(dims[0])
        self.conv1_2 = BasicBlock(dims[1])
        self.conv2_2 = BasicBlock(dims[2])

        self.conv0_3 = BasicBlock(dims[0])
        self.conv1_3 = BasicBlock(dims[1])

        self.conv0_4 = BasicBlock(dims[0])
        self.up = layers.UpSampling2D()

    def call(self, x, training=None):
        x0_0, x1_0, x2_0, x3_0, x4_0 = x

        x0_1 = self.conv0_1(layers.concatenate([x0_0, self.up(x1_0)], -1))
        x1_1 = self.conv1_1(layers.concatenate([x1_0, self.up(x2_0)], -1))
        x0_2 = self.conv0_2(layers.concatenate([x0_0, x0_1, self.up(x1_1)], -1))

        x2_1 = self.conv2_1(layers.concatenate([x2_0, self.up(x3_0)], -1))
        x1_2 = self.conv1_2(layers.concatenate([x1_0, x1_1, self.up(x2_1)], -1))
        x0_3 = self.conv0_3(layers.concatenate([x0_0, x0_1, x0_2, self.up(x1_2)], -1))

        x3_1 = self.conv3_1(layers.concatenate([x3_0, self.up(x4_0)], -1))
        x2_2 = self.conv2_2(layers.concatenate([x2_0, x2_1, self.up(x3_1)], -1))
        x1_3 = self.conv1_3(layers.concatenate([x1_0, x1_1, x1_2, self.up(x2_2)], -1))
        x0_4 = self.conv0_4(layers.concatenate([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], -1))

        return x0_4


class Head(Sequential):
    def __init__(self, cout):
        """
        Basic Identity
        :param int cout: output channel
        """
        conv = layers.Conv2D(cout, 1)
        super().__init__(conv)


if __name__ == '__main__':
    H = W = 224
    dim = 32
    num_cls = 10
    inputs = tf.random.normal((1, H, W, 3))

    encoder = Encoder(base_dim=dim)
    outputs = encoder(inputs, training=True)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, H // (2 ** i), W // (2 ** i), dim * (2 ** i)]

    bridger = Bridger()
    outputs = bridger(outputs, training=True)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, H // (2 ** i), W // (2 ** i), dim * (2 ** i)]

    features = [tf.random.normal((1, H // (2 ** i), W // (2 ** i), dim * (2 ** i))) for i in range(5)]
    decoder = Decoder(dim)
    decoder.build([f.shape for f in features])
    outputs = decoder(features, training=True)
    assert list(outputs.shape) == [1, H, W, dim]

    head = Head(num_cls)
    outputs = head(outputs, training=True)
    assert list(outputs.shape) == [1, H, W, num_cls]

    unet = TFUNet2plus(num_cls, dim)
    outputs = unet(inputs, training=True)
    assert list(outputs.shape) == [1, H, W, num_cls]

