# You won't directly use this class, but it will be used by the other classes.
# Unless you want to build the class yourself, you may use this

import torch.nn as nn

class CNNClassification(nn.Module):
    """
    Base PyTorch class of the classification model with Encoder, Head for CNN
    """
    def forward(self, x):
        x_T = x.transpose(-1,-2) # Transpose [BATCH, TIME, CHANNEL] => [BATCH, CHANNEL, TIME]
        features = self.encoder(x_T) # => [BATCH, 512, TIME]
        features_agg = self.globalpooling(features) # => [BATCH, 512, 1]
        output = self.head(features_agg[:, :, -1])
        return output

class RNNClassification(nn.Module):
    """
    Base PyTorch class of the classification model with Encoder, Head for RNN
    """
    def forward(self, x):
        features, _ = self.encoder(x) # RNN will output feature and states
        output = self.head(features[:, -1, :])
        return output