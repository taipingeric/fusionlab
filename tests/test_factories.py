import torch
from fusionlab.layers.factories import (
    ConvND, ConvT, Upsample, BatchNorm,
    MaxPool, AvgPool, AdaptiveMaxPool, AdaptiveAvgPool,
    ReplicationPad, ConstantPad
)



# Test Code for ConvND
inputs = torch.randn(1, 3, 16) # create random input tensor
layer = ConvND(spatial_dims=1, in_channels=3, out_channels=2, kernel_size=5) # create model instance
outputs = layer(inputs) # pass input through model
print(outputs.shape)
assert list(outputs.shape) == [1, 2, 12] # check output shape is correct

# Test code for ConvT
inputs = torch.randn(1, 3, 16) # create random input tensor
layer = ConvT(spatial_dims=1, in_channels=3, out_channels=2, kernel_size=5) # create model instance
outputs = layer(inputs) # pass input through model
print(outputs.shape)
assert list(outputs.shape) == [1, 2, 20] # check output shape is correct

# Test code for Upsample
inputs = torch.randn(1, 3, 16) # create random input tensor
layer = Upsample(spatial_dims=1, scale_factor=2) # create model instance
outputs = layer(inputs) # pass input through model
print(outputs.shape)
assert list(outputs.shape) == [1, 3, 32] # check output shape is correct

# Test code for BatchNormND
inputs = torch.randn(1, 3, 16) # create random input tensor
layer = BatchNorm(spatial_dims=1, num_features=3) # create model instance
outputs = layer(inputs) # pass input through model

# Test code for MaxPool
for Module in [MaxPool, AvgPool]:
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = Module(spatial_dims=1, kernel_size=2) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 3, 8] # check output shape is correct

# Test code for Pool
for Module in [AdaptiveMaxPool, AdaptiveAvgPool]:
    inputs = torch.randn(1, 3, 16) # create random input tensor
    layer = Module(spatial_dims=1, output_size=8) # create model instance
    outputs = layer(inputs) # pass input through model
    print(outputs.shape)
    assert list(outputs.shape) == [1, 3, 8] # check output shape is correct

# Test code for Padding
inputs = torch.randn(1, 3, 16) # create random input tensor
layer = ReplicationPad(spatial_dims=1, padding=2) # create model instance
outputs = layer(inputs) # pass input through model
print(outputs.shape)
assert list(outputs.shape) == [1, 3, 20] # check output shape is correct

layer = ConstantPad(spatial_dims=1, padding=2, value=0) # create model instance
outputs = layer(inputs) # pass input through model
print(outputs.shape)
assert list(outputs.shape) == [1, 3, 20] # check output shape is correct
