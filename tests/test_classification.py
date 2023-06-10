import torch
from fusionlab.classification import HFClassificationModel
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