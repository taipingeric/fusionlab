# You won't directly use this class, but it will be used by the other classes.
# Unless you want to build the class yourself, you may use this

import torch.nn as nn

class CNNClassificationModel(nn.Module):
    """
    Base PyTorch class of the classification model with Encoder, Head for CNN
    """
    def forward(self, x):
        # 1D signal x => [BATCH, CHANNEL, TIME]
        # 1D spectrum x => [BATCH, FREQUENCY, TIME] (single channel)
        # 2D spectrum x => [BATCH, CHANNEL, FREQUENCY, TIME] (multi channel)
        # 2D image x => [BATCH, CHANNEL, HEIGHT, WIDTH]
        # 3D volumetric x => [BATCH, CHANNEL, HEIGHT, WIDTH, DEPTH]
        features = self.encoder(x) # => [BATCH, 512, ...]
        features_agg = self.globalpooling(features) # => [BATCH, 512, 1, (1, (1))]
        output = self.head(features_agg.view(x.shape[0],-1)) # => [BATCH, NUM_CLS]
        return output

class RNNClassificationModel(nn.Module):
    """
    Base PyTorch class of the classification model with Encoder, Head for RNN
    """
    def forward(self, x):
        # 1D signal x => [BATCH, CHANNEL, TIME]
        x = x.transpose(1,2)
        features, _ = self.encoder(x) # RNN will output feature and states
        output = self.head(features[:, -1, :])
        return output

class HFClassificationModel(nn.Module):
    """
    Base Hugginface-pytoch model wrapper class of the classification model
    """
    def __init__(self, model,
                 num_cls=None,
                 loss_fct=nn.CrossEntropyLoss()):
        super().__init__()
        self.net = model
        if 'num_cls' in model.__dict__.keys():
            self.num_cls = model.num_cls
        else:
            self.num_cls = num_cls
        assert self.num_cls is not None, "num_cls is not defined"
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
    from fusionlab.classification import LSTMClassifier

    H = W = 224
    cout = 5
    inputs = torch.normal(0, 1, (1, 3, W))
    # Test CNNClassification
    model = VGG16Classifier(3, cout, spatial_dims=1)
    hf_model = HFClassificationModel(model, cout)
    output = hf_model(inputs)
    print(output['logits'].shape)
    assert list(output.keys()) == ['loss', 'logits', 'hidden_states']

    inputs = torch.normal(0, 1, (1, 3, H, W))
    # Test CNNClassification
    model = VGG16Classifier(3, cout, spatial_dims=2)
    hf_model = HFClassificationModel(model, cout)
    output = hf_model(inputs)
    print(output['logits'].shape)
    assert list(output.keys()) == ['loss', 'logits', 'hidden_states']

    inputs = torch.normal(0, 1, (1, 3, H))
    model = LSTMClassifier(3, cout)
    hf_model = HFClassificationModel(model, cout)
    output = hf_model(inputs)
    print(output['logits'].shape)
    assert list(output.keys()) == ['loss', 'logits', 'hidden_states']