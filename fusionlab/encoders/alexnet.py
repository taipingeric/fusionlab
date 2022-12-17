import torch
import torch.nn as nn
import tensorflow as tf

# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
class AlexNet(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# TF
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
    inputs = torch.normal(0, 1, (1, 3, 224, 224))
    output = AlexNet(3)(inputs)
    pt_shape = output.shape

    inputs = tf.random.normal((1, 224, 224, 3))
    output = TFAlexNet()(inputs)
    tf_shape = output.shape

    assert tf_shape[1:3] == pt_shape[-2:] and pt_shape[1] == tf_shape[-1]
