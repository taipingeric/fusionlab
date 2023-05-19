from torch import nn

Conv = {
    1:nn.Conv1d,
    2:nn.Conv2d,
    3:nn.Conv3d
}
ConvT = {
    1:nn.ConvTranspose1d,
    2:nn.ConvTranspose2d,
    3:nn.ConvTranspose3d
}

BatchNorm = {
    1:nn.BatchNorm1d,
    2:nn.BatchNorm2d,
    3:nn.BatchNorm3d
}

MaxPool = {
    1:nn.MaxPool1d,
    2:nn.MaxPool2d,
    3:nn.MaxPool3d
}

AdaptiveMaxPool = {
    1:nn.AdaptiveMaxPool1d,
    2:nn.AdaptiveMaxPool2d,
    3:nn.AdaptiveMaxPool3d
}

AvgPool = {
    1:nn.AvgPool1d,
    2:nn.AvgPool2d,
    3:nn.AvgPool3d
}

AdaptiveAvgPool = {
    1:nn.AdaptiveAvgPool1d,
    2:nn.AdaptiveAvgPool2d,
    3:nn.AdaptiveAvgPool3d
}

ReplicationPad = {
    1:nn.ReplicationPad1d,
    2:nn.ReplicationPad2d,
    3:nn.ReplicationPad3d
}

ConstantPad = {
    1:nn.ConstantPad1d,
    2:nn.ConstantPad2d,
    3:nn.ConstantPad3d
}