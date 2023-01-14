# FusionLab

<p align="center">
    <br>
    <img src="assets/imgs/fusionlab_banner.png" width="400"/>
    <br>
<p>

FusionLab is an open-source frameworks built for Deep Learning research written in PyTorch and Tensorflow. The code is easy to read and modify 
especially for newbie. Feel free to send pull requests :D

* [Installation](#Installation)
* [How to use](#How-to-use)
* [Encoders](#Encoders)
* [Segmentation](#Segmentation)

## Installation

### With pip

```bash
pip install fusionlab
```

#### For Mac M1 chip users

Requirements:
* Apple Mac with M1 chips
* MacOS > 12.6 (Monterey)

Following steps
1. Clone this repo
```bash
git clone https://github.com/taipingeric/fusionlab.git
cd fusionlab
```
2. (remove anaconda first)
3. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   1. Miniconda3 macOS Apple M1 64-bit pkg
   2. Miniconda3 macOS Apple M1 64-bit bash
4. Install the xcode-select command-line
```bash
xcode-select --install
```
5. Deactivate the base environment
```bash
conda deactivate 
```
6. Create conda environment using [config](./configs/tf-apple-m1-conda.yaml)
```bash
conda env create -f ./configs/tf-apple-m1-conda.yaml -n fusionlab
```
7. Replace [requirements.txt](requirements.txt) with [requirements-m1.txt](configs/requirements-m1.txt)
8. Install by pip
```bash
pip install -r requirements-m1.txt
```

ref: [https://github.com/jeffheaton/t81_558_deep_learning/install/tensorflow-install-conda-mac-metal-jan-2023.ipynb](https://github.com/jeffheaton/t81_558_deep_learning/install/tensorflow-install-conda-mac-metal-jan-2023.ipynb)

[video](https://www.youtube.com/watch?v=5DgWvU0p2bk) 

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

## Segmentation

```python
import fusionlab as fl

# Segmentation Model

# PyTorch UNet
unet = fl.segmentation.UNet(cin=3, num_cls=10, base_dim=64)
# Tensorflow UNet
unet = fl.segmentation.TFUNet(num_cls=10, base_dim=64)

```

[Segmentation model list](fusionlab/segmentation/README.md)

* UNet, TFUNet
* ResUNet, TFResUNet
* UNet2plus, TFUNet2plus
