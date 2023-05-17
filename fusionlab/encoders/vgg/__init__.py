from fusionlab.configs import BACKEND
if BACKEND == 'torch':
    from .vgg import VGG16, VGG19
elif BACKEND == 'tf':
    from .tfvgg import TFVGG16, TFVGG19
else:
    print('backend not supported!!!')