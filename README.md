# FusionLab

<p align="center">
    <br>
    <img src="assets/imgs/fusionlab_banner.png" width="400"/>
    <br>
<p>

![Test](https://github.com/taipingeric/fusionlab/actions/workflows/python-app.yml/badge.svg)  [![Downloads](https://static.pepy.tech/badge/fusionlab)](https://pepy.tech/project/fusionlab)

FusionLab is an open-source frameworks built for Deep Learning research written in PyTorch and Tensorflow. The code is easy to read and modify 
especially for newbie. Feel free to send pull requests :D

* [What's New](#News)
* [Installation](#Installation)
* [How to use](#How-to-use)
* [Encoders](#Encoders)
* [Losses](#Losses)
* [Segmentation](#Segmentation)
* [Acknowledgements](#Acknowledgements)

## Installation

### With pip

```bash
pip install fusionlab
```

#### For Mac M1 chip users
Go to [Install on Macbook M1 chip](./configs/Install%20on%20Macbook%20M1.md) 

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
import torch
import tensorflow as tf

# PyTorch
pred = torch.normal(0., 1., (1, 3, 4, 4)) # (N, C, *)
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
pred = torch.normal(0, 1, (1, 1, 4, 4)) # (N, 1, *)
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
unet = fl.segmentation.UNet(cin=3, num_cls=10, base_dim=64)

# Tensorflow UNet
import tensorflow as tf

# Multiclass Segmentation
unet = tf.keras.Sequential([
   fl.segmentation.TFUNet(num_cls=10, base_dim=64),
   tf.keras.layers.Activation(tf.nn.softmax),
])
unet.compile(loss=fl.losses.TFDiceLoss("multiclass"))

# Binary Segmentation
unet = tf.keras.Sequential([
   fl.segmentation.TFUNet(num_cls=1, base_dim=64),
   tf.keras.layers.Activation(tf.nn.sigmoid),
])
unet.compile(loss=fl.losses.TFDiceLoss("binary"))


```

[Segmentation model list](fusionlab/segmentation/README.md)

* UNet
* ResUNet
* UNet2plus

## News

0.0.52

* Tversky Loss for Torch and TF

## Acknowledgements

* [BloodAxe/pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
