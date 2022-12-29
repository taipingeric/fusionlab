import torch
import torch.nn as nn


# Official pytorch ref: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
class AlexNet(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)



if __name__ == '__main__':
    img_size = 224
    inputs = torch.normal(0, 1, (1, 3, img_size, img_size))
    output = AlexNet(3)(inputs)
    shape = list(output.shape)
    print(shape)
    assert shape[-2:] == [6, 6]
