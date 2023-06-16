import torch
from torch import nn
from fusionlab.classification.base import RNNClassificationModel


class LSTMClassifier(RNNClassificationModel):
    def __init__(self, cin, cout, hidden_size=512):
        super().__init__()
        self.encoder = nn.LSTM(input_size=cin, hidden_size=hidden_size, batch_first=True) # define LSTM layer
        self.head = nn.Linear(hidden_size, cout) # define output head layer

if __name__ == '__main__':
    inputs = torch.randn(1, 5, 3) # create random input tensor
    model = LSTMClassifier(cin=3, hidden_size=4, cout=2) # create model instance
    outputs = model(inputs) # pass input through model
    assert list(outputs.shape) == [1, 2] # check output shape is correct