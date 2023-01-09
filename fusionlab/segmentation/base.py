import torch.nn as nn
from tensorflow.keras import Model


class SegmentationModel(nn.Module):
    """
    Base PyTorch class of the segmentation model with Encoder, Bridger, Decoder, Head
    """
    def forward(self, x):
        features = self.encoder(x)
        feature_fusion = self.bridger(features)
        decoder_output = self.decoder(feature_fusion)
        output = self.head(decoder_output)
        return output


class TFSegmentationModel(Model):
    """
    Base PyTorch class of the segmentation model with Encoder, Bridger, Decoder, Head
    """
    def call(self, x):
        features = self.encoder(x)
        feature_fusion = self.bridger(features)
        decoder_output = self.decoder(feature_fusion)
        output = self.head(decoder_output)
        return output
