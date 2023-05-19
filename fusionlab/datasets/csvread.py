# Reader for csv files
import numpy as np

__all__ = ['read_csv']

def read_csv(fname):
    return np.genfromtxt(fname, dtype=np.float32, skip_header=1, delimiter=",")