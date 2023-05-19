from tensorflow.keras import Model

class TFSegmentationModel(Model):
    """
    Base PyTorch class of the segmentation model with Encoder, Bridger, Decoder, Head
    """
    def call(self, x, training=None):
        """

        Args:
            x: input tensor
            training: flag for BatchNormalization and Dropout, whether the layer should behave in training mode or in
            inference mode

        Returns:

        """
        features = self.encoder(x, training)
        feature_fusion = self.bridger(features, training)
        decoder_output = self.decoder(feature_fusion, training)
        output = self.head(decoder_output, training)
        return output
