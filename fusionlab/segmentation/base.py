import torch.nn as nn


class SegmentationModel(nn.Module):
    def forward(self, x):
        features = self.encoder(x)
        feature_fusion = self.bridger(features)
        decoder_output = self.decoder(feature_fusion)
        output = self.head(decoder_output)
        return output
