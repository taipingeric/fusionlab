import matplotlib.pyplot as plt

def plot_channels(signals):
    '''
    plot signals by channels

    Args:
        signals: numpy array, shape (num_samples, num_channels)
    '''
    num_channels = signals.shape[1]
    _, axes = plt.subplots(num_channels, 1, figsize=(10, 10))
    for i in range(num_channels):
        axes[i].plot(signals[:, i])
    plt.show()
