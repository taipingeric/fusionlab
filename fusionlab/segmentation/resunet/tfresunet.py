import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from fusionlab.segmentation.tfbase import TFSegmentationModel


class TFResUNet(TFSegmentationModel):
    def __init__(self, num_cls, base_dim):
        super().__init__()
        self.encoder = Encoder(base_dim)
        self.bridger = Bridger()
        self.decoder = Decoder(base_dim)
        self.head = Head(num_cls)


class Encoder(Model):
    def __init__(self, base_dim):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(4)]
        self.stem = Stem(dims[0])
        self.stage1 = ResConv(dims[1], stride=2)
        self.stage2 = ResConv(dims[2], stride=2)
        self.stage3 = ResConv(dims[3], stride=2)

    def call(self, x, training):
        s0 = self.stem(x, training)
        s1 = self.stage1(s0, training)
        s2 = self.stage2(s1, training)
        s3 = self.stage3(s2, training)
        return [s0, s1, s2, s3]


class Decoder(Model):
    def __init__(self, base_dim):
        """
        Base UNet decoder
        Args:
            base_dim (int): output dim of deepest stage output or input channels
        """
        super().__init__()
        dims = [base_dim*(2**i) for i in range(4)]
        self.d3 = DecoderBlock(dims[2])
        self.d2 = DecoderBlock(dims[1])
        self.d1 = DecoderBlock(dims[0])

    def call(self, x, training):
        s0, s1, s2, s3 = x

        x = self.d3(s3, s2, training)
        x = self.d2(x, s1, training)
        x = self.d1(x, s0, training)
        return x


class DecoderBlock(Model):
    def __init__(self, cout):
        super().__init__()
        self.upsample = layers.Conv2DTranspose(cout, 2, strides=2)
        self.conv = ResConv(cout, stride=1)

    def call(self, x1, x2, training):
        x1 = self.upsample(x1)
        x = tf.concat([x1, x2], axis=-1)
        return self.conv(x)


class Bridger(Model):
    def __init__(self):
        super().__init__()

    def call(self, x, training):
        outputs = [tf.identity(i) for i in x]
        return outputs


class Stem(Model):
    def __init__(self, cout):
        super().__init__()
        self.conv = Sequential([
            layers.Conv2D(cout, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(cout, 3, padding='same'),
        ])
        self.skip = Sequential(
            layers.Conv2D(cout, 3, padding='same'),
        )

    def call(self, x, training):
        return self.conv(x) + self.skip(x)


class ResConv(Model):
    def __init__(self, cout, stride=1):
        super().__init__()

        self.conv = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(cout, 3, stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(cout, 3, padding='same'),
        ])
        self.skip = Sequential([
            layers.Conv2D(cout, 3, strides=stride, padding='same'),
            layers.BatchNormalization(),
        ])

    def call(self, x, training=None):
        return self.conv(x, training) + self.skip(x, training)


class Head(Sequential):
    def __init__(self, cout):
        """
        Basic conv head
        :param int cout: number of classes
        """
        conv = layers.Conv2D(cout, 1)
        super().__init__(conv)


if __name__ == '__main__':
    H = W = 224
    cout = 64
    inputs = tf.random.normal((1, H, W, 3))

    model = TFResUNet(100, cout)
    output = model(inputs, training=True)
    print(output.shape)

    dblock = DecoderBlock(128)
    inputs2 = tf.random.normal((1, H, W, 128))
    inputs1 = tf.random.normal((1, H//2, W//2, 64))
    outputs = dblock(inputs1, inputs2, training=True)
    print(outputs.shape)

    encoder = Encoder(cout)
    outputs = encoder(inputs, training=True)
    for o in outputs:
        print(o.shape)

    decoder = Decoder(cout)
    outputs = decoder(outputs, training=True)
    print("Encoder + Decoder ", outputs.shape)

    stem = Stem(cout)
    outputs = stem(inputs, training=True)
    print(outputs.shape)
    assert list(outputs.shape) == [1, H, W, cout]

    resconv = ResConv(cout, 1)
    outputs = resconv(inputs, training=True)
    print(outputs.shape)
    assert list(outputs.shape) == [1, H, W, cout]

    resconv = ResConv(cout, stride=2)
    outputs = resconv(inputs, training=True)
    print(outputs.shape)
    assert list(outputs.shape) == [1, H//2, W//2, cout]