from fusionlab import BACKEND

if BACKEND['torch']:
    from .base import (
        CNNClassificationModel, 
        RNNClassificationModel, 
        HFClassificationModel
    )
    from .lstm import LSTMClassifier
    from .vgg import (
        VGG16Classifier, 
        VGG19Classifier
    )
    from .base import HFClassificationModel