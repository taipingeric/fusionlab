# VGG Classifier
import torch
from torch import nn
from fusionlab.classification.base import CNNClassificationModel
from fusionlab.encoders import VGG16, VGG19
from fusionlab.layers import AdaptiveAvgPool

class VGG16Classifier(CNNClassificationModel):
    def __init__(self, cin, num_cls, spatial_dims=2):
        super().__init__()
        self.num_cls = num_cls
        self.encoder = VGG16(cin, spatial_dims) # Create VGG16 instance
        self.globalpooling = AdaptiveAvgPool(spatial_dims, 1)
        self.head = nn.Linear(512, num_cls)

class VGG19Classifier(CNNClassificationModel):
    def __init__(self, cin, num_cls, spatial_dims=2):
        super().__init__()
        self.num_cls = num_cls
        self.encoder = VGG19(cin, spatial_dims) # Create VGG16 instance
        self.globalpooling = AdaptiveAvgPool(spatial_dims, 1)
        self.head = nn.Linear(512, num_cls)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224) # create random input tensor
    model = VGG16Classifier(cin=3, num_cls=2, spatial_dims=1) # create model instance
    outputs = model(inputs) # pass input through model
    assert list(outputs.shape) == [1, 2] # check output shape is correct

    inputs = torch.randn(1, 3, 224) # create random input tensor
    model = VGG19Classifier(cin=3, num_cls=2, spatial_dims=1) # create model instance
    
    outputs = model(inputs) # pass input through model
    assert list(outputs.shape) == [1, 2] # check output shape is correct