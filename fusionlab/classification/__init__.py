from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .base import CNNClassification, RNNClassification, HFClassification
    from .lstm import LSTMClassifier
    from .vgg import VGG16Classifier, VGG19Classifier
    from .base import HFSegmentationModel
elif BACKEND == 'tf':
    print('not built yet')
else:
    print('backend not supported!!!')

    
