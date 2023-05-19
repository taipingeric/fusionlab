import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from fusionlab.segmentation.tfbase import TFSegmentationModel


class TFUNet(TFSegmentationModel):
    def __init__(self, num_cls, base_dim=64):
        """
        Base Unet
        Args:
            num_cls (int): number of classes
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        stage = 5
        self.encoder = Encoder(base_dim)
        self.bridger = Bridger()
        self.decoder = Decoder(base_dim*(2**(stage-2)))  # 512
        self.head = Head(num_cls)


class Encoder(Model):
    def __init__(self, base_dim):
        """
        UNet Encoder
        Args:
            base_dim (int): 1st stage dim of conv output
        """
        super().__init__()
        self.pool = layers.MaxPool2D()
        self.stage1 = BasicBlock(base_dim)
        self.stage2 = BasicBlock(base_dim * 2)
        self.stage3 = BasicBlock(base_dim * 4)
        self.stage4 = BasicBlock(base_dim * 8)
        self.stage5 = BasicBlock(base_dim * 16)

    def call(self, x, training=None):
        s1 = self.stage1(x, training)
        x = self.pool(s1)
        s2 = self.stage2(x, training)
        x = self.pool(s2)
        s3 = self.stage3(x, training)
        x = self.pool(s3)
        s4 = self.stage4(x, training)
        x = self.pool(s4)
        s5 = self.stage5(x, training)

        return [s1, s2, s3, s4, s5]


class Decoder(Model):
    def __init__(self, base_dim):
        """
        Base UNet decoder
        Args:
            base_dim (int): output dim of deepest stage output
        """
        super().__init__()
        self.d4 = DecoderBlock(base_dim)
        self.d3 = DecoderBlock(base_dim//2)
        self.d2 = DecoderBlock(base_dim//4)
        self.d1 = DecoderBlock(base_dim//8)

    def call(self, x, training=None):
        f1, f2, f3, f4, f5 = x
        x = self.d4(f5, f4, training)
        x = self.d3(x, f3, training)
        x = self.d2(x, f2, training)
        x = self.d1(x, f1, training)
        return x


class Bridger(Model):
    def __init__(self):
        super().__init__()

    def call(self, x, training=None):
        outputs = [tf.identity(i) for i in x]
        return outputs


class Head(Sequential):
    def __init__(self, cout):
        """
        Basic Identity
        :param int cout: output channel
        """
        conv = layers.Conv2D(cout, 1)
        super().__init__(conv)


class BasicBlock(Sequential):
    def __init__(self, cout):
        conv1 = Sequential([
            layers.Conv2D(cout, 3, 1, padding='same'),
            layers.ReLU(),
        ])
        conv2 = Sequential([
            layers.Conv2D(cout, 3, 1, padding='same'),
            layers.ReLU(),
        ])
        super().__init__([conv1, conv2])


class DecoderBlock(Model):
    def __init__(self, cout):
        """
        Base Unet decoder block for merging the outputs from 2 stages
        Args:
            cout: output dim of the block
        """
        super().__init__()
        self.up = layers.UpSampling2D()
        self.conv = BasicBlock(cout)

    def call(self, x1, x2, training=None):
        x1 = self.up(x1)
        x = tf.concat([x1, x2], axis=-1)
        x = self.conv(x, training)
        return x


if __name__ == '__main__':
    H = W = 224
    dim = 64
    inputs = tf.random.normal((1, H, W, 3))

    encoder = Encoder(dim)
    encoder.build((1, H, W, 3))
    outputs = encoder(inputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, H // (2 ** i), W // (2 ** i), dim * (2 ** i)]

    bridger = Bridger()
    outputs = bridger(outputs)
    for i, o in enumerate(outputs):
        assert list(o.shape) == [1, H // (2 ** i), W // (2 ** i), dim * (2 ** i)]

    features = [tf.random.normal((1, H // (2 ** i), W // (2 ** i), dim * (2 ** i))) for i in range(5)]
    decoder = Decoder(512)
    outputs = decoder(features)
    assert list(outputs.shape) == [1, H, W, 64]

    head = Head(10)
    outputs = head(outputs)
    assert list(outputs.shape) == [1, H, W, 10]

    unet = TFUNet(10)
    outputs = unet(inputs)
    assert list(outputs.shape) == [1, H, W, 10]
