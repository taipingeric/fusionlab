# Reader for csv files
import numpy as np

def read_csv(fname):
    readout = np.genfromtxt(fname, dtype=np.float32, skip_header=1, delimiter=",")
    return readout[:, 1:]

if __name__ == "__main__":
    signal = read_csv('csv_sample.csv')
    assert list(signal.shape) == [3, 3]
    assert list(signal[:,0]) == [1., 2., 3.]