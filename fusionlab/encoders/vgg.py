import torch
import torch.nn as nn
import tensorflow as tf


# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
class VGG16(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class VGG19(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        ksize = 3
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


# TF
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
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG16(3)(inputs)
    pt_shape = output.shape

    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFVGG16()(inputs)
    tf_shape = output.shape
    print(pt_shape, tf_shape)
    assert tf_shape[1:3] == pt_shape[-2:] and pt_shape[1] == tf_shape[-1]
    # VGG19
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = VGG19(3)(inputs)
    pt_shape = output.shape

    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFVGG19()(inputs)
    tf_shape = output.shape
    print(pt_shape, tf_shape)
    assert tf_shape[1:3] == pt_shape[-2:] and pt_shape[1] == tf_shape[-1]