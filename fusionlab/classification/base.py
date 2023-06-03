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

class HFClassification(nn.Module):
    """
    Base Hugginface-pytoch model wrapper class of the classification model
    """
    def __init__(self, model,
                 num_cls=4,
                 loss_fct=nn.CrossEntropyLoss()):
        self.net = model
        self.num_cls = num_cls
        self.loss_fct = loss_fct
    def forward(self, x, labels=None):
        logits = self.net(x)  # Forward pass the model
        if labels is not None:
            # logits => [BATCH, NUM_CLS]
            # labels => [BATCH]
            loss = self.loss_fct(logits.view(-1, self.num_cls), labels.view(-1))  # Calculate loss
        else:
            loss = None
        # return dictionary for hugginface trainer
        return {'loss':loss, 'logits':logits, 'hidden_states':None}

# Test the function
if __name__ == '__main__':
    import torch
    from fusionlab.classification import VGG16Classifier
    H = W = 224
    cout = 5
    inputs = torch.normal(0, 1, (1, 3, H, W))

    # Test CNNClassification
    model = VGG16Classifier(spatial_dims=3, 3, cout, 64)
    hf_model = HFSegmentationModel(model, cout)
    output = hf_model(inputs)
    assert list(output.keys()) == ['loss', 'logits', 'hidden_states']