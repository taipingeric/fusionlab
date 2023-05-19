# VGG Classifier
import torch
from torch import nn
from ai4ecg.classification.base import CNNClassification
from ai4ecg.encoder import vgg

class VGG16Classifier(CNNClassification):
    def __init__(self, cin, cout):
        super().__init__()
        self.encoder = vgg.VGG16(cin) # Create VGG16 instance
        self.head = nn.Linear(512, cout)

class VGG19Classifier(CNNClassification):
    def __init__(self, cin, cout):
        super().__init__()
        self.encoder = vgg.VGG19(cin) # Create VGG19 instance
        self.head = nn.Linear(512, cout)


if __name__ == '__main__':
    inputs = torch.randn(1, 224, 3) # create random input tensor
    model = VGG16Classifier(cin=3, cout=2) # create model instance
    outputs = model(inputs) # pass input through model
    assert list(outputs.shape) == [1, 2] # check output shape is correct

    inputs = torch.randn(1, 224, 3) # create random input tensor
    model = VGG19Classifier(cin=3, cout=2) # create model instance
    outputs = model(inputs) # pass input through model
    assert list(outputs.shape) == [1, 2] # check output shape is correct