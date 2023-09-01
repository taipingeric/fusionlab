import collections
from itertools import repeat
from typing import Any, Tuple

def autopad(kernel_size, padding=None, dilation=1, spatial_dims=2):
    '''
    Auto padding for convolutional layers
    '''
    if padding is None:
        if isinstance(kernel_size, int) and isinstance(dilation, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            kernel_size = make_ntuple(kernel_size, spatial_dims)
            dilation = make_ntuple(dilation, spatial_dims)
            padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(spatial_dims))
    return padding

def make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/vision/blob/main/torchvision/utils.py#L585C1-L597C31

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))