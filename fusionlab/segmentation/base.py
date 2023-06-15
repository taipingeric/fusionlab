import torch.nn as nn

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
    
class HFSegmentationModel(nn.Module):
    """
    Base Hugginface-pytoch model wrapper class of the segmentation model
    """
    def __init__(self, model, num_cls=None,
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
            loss = self.loss_fct(logits, labels)  # Calculate loss
        else:
            loss = None
        # return dictionary for hugginface trainer
        return {'loss':loss, 'logits':logits, 'hidden_states':None}


if __name__ == '__main__':
    import torch
    from fusionlab.segmentation import ResUNet
    H = W = 224
    cout = 5
    inputs = torch.normal(0, 1, (1, 3, H, W))

    model = ResUNet(3, cout, 64)
    hf_model = HFSegmentationModel(model, cout)
    output = hf_model(inputs)
    assert list(output.keys()) == ['loss', 'logits', 'hidden_states']
    print(output['logits'].shape)
    assert list(output['logits'].shape) == [1, cout, H, W]