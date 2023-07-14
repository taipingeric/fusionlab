import collections
from itertools import repeat
from typing import Any, Tuple

def autopad(kernal_size, padding=None):
    # Pad to 'same'
    if padding is None:
        padding = kernal_size//2 if isinstance(kernal_size, int) else [x//2 for x in kernal_size]
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