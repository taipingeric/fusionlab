# FusionLab

<p align="center">
    <br>
    <img src="assets/imgs/fusionlab_banner.png" width="400"/>
    <br>
<p>

[![PyPI version](https://badge.fury.io/py/fusionlab.svg)](https://badge.fury.io/py/fusionlab) ![Test](https://github.com/taipingeric/fusionlab/actions/workflows/python-app.yml/badge.svg)  [![Downloads](https://static.pepy.tech/badge/fusionlab)](https://pepy.tech/project/fusionlab)

FusionLab is an open-source frameworks built for Deep Learning research written in PyTorch and Tensorflow. The code is easy to read and modify 
especially for newbie. Feel free to send pull requests :D

* [What's New](#News)
* [Installation](#Installation)
* [How to use](#How-to-use)
* [Encoders](#Encoders)
* [Losses](#Losses)
* [Segmentation](#Segmentation)
* [1D, 2D, 3D Model](#n-dimensional-model)
* [Acknowledgements](#Acknowledgements)

## Installation

### With pip

```bash
pip install fusionlab
```

#### For Mac M1 chip users
[Install on Macbook M1 chip](./configs/Install%20on%20Macbook%20M1.md) 

## How to use

```python
import fusionlab as fl

# PyTorch
encoder = fl.encoders.VGG16()
# Tensorflow
encoder = fl.encoders.TFVGG16()

```

## Encoders

[encoder list](fusionlab/encoders/README.md)

## Losses

[Loss func list](fusionlab/losses/README.md)
* Dice Loss
* Tversky Loss
* IoU Loss


```python
# Dice Loss (Multiclass)
import fusionlab as fl

# PyTorch
pred = torch.randn(1, 3, 4, 4) # (N, C, *)
target = torch.randint(0, 3, (1, 4, 4)) # (N, *)
loss_fn = fl.losses.DiceLoss()
loss = loss_fn(pred, target)

# Tensorflow
pred = tf.random.normal((1, 4, 4, 3), 0., 1.) # (N, *, C)
target = tf.random.uniform((1, 4, 4), 0, 3) # (N, *)
loss_fn = fl.losses.TFDiceLoss("multiclass")
loss = loss_fn(target, pred)


# Dice Loss (Binary)

# PyTorch
pred = torch.randn(1, 1, 4, 4) # (N, 1, *)
target = torch.randint(0, 3, (1, 4, 4)) # (N, *)
loss_fn = fl.losses.DiceLoss("binary")
loss = loss_fn(pred, target)

# Tensorflow
pred = tf.random.normal((1, 4, 4, 1), 0., 1.) # (N, *, 1)
target = tf.random.uniform((1, 4, 4), 0, 3) # (N, *)
loss_fn = fl.losses.TFDiceLoss("binary")
loss = loss_fn(target, pred)


```

## Segmentation

```python
import fusionlab as fl
# PyTorch UNet
unet = fl.segmentation.UNet(cin=3, num_cls=10)

# Tensorflow UNet
# Multiclass Segmentation
unet = tf.keras.Sequential([
   fl.segmentation.TFUNet(num_cls=10, base_dim=64),
   tf.keras.layers.Activation(tf.nn.softmax),
])

# Binary Segmentation
unet = tf.keras.Sequential([
   fl.segmentation.TFUNet(num_cls=1, base_dim=64),
   tf.keras.layers.Activation(tf.nn.sigmoid),
])
```

[Segmentation model list](fusionlab/segmentation/README.md)

* UNet
* ResUNet
* UNet2plus

## N Dimensional Model

some models can be used in 1D, 2D, 3D

```python
import fusionlab as fl

resnet1d = fl.encoders.ResNet50V1(cin=3, spatial_dims=1)
resnet2d = fl.encoders.ResNet50V1(cin=3, spatial_dims=2)
resnet3d = fl.encoders.ResNet50V1(cin=3, spatial_dims=3)

unet1d = fl.segmentation.UNet(cin=3, num_cls=10, spatial_dims=1)
unet2d = fl.segmentation.UNet(cin=3, num_cls=10, spatial_dims=2)
unet3d = fl.segmentation.UNet(cin=3, num_cls=10, spatial_dims=3)
```

## News

[Release logs](./release_logs.md)

## Acknowledgements

* [BloodAxe/pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
