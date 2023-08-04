from .__version__ import __version__

# Check if PyTorch or TensorFlow is installed
try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False

try:
    import tensorflow as tf
    tf_installed = True
except ImportError:
    tf_installed = False

print(f"PyTorch installed: {torch_installed}")
print(f"TensorFlow installed: {tf_installed}")

BACKEND = {
    "torch": torch_installed,
    "tf": tf_installed
}

# check if no backend installed
if not any(BACKEND.values()):
    print("None of supported backend installed")

from . import (
    functional,
    encoders,
    utils,
    datasets,
    layers,
    classification,
    segmentation,
    losses,
    trainers,
    configs
)