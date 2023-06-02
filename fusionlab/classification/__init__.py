from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .base import CNNClassification, RNNClassification
    from .lstm import LSTMClassifier
    from .vgg import VGG16Classifier, VGG19Classifier
elif BACKEND == 'tf':
    print('not built yet')
else:
    print('backend not supported!!!')

    
