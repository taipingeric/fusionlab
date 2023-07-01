import matplotlib.pyplot as plt

def plot_channels(signals, show=True):
    '''
    plot signals by channels

    Args:
        signals: numpy array, shape (num_samples, num_channels)
    '''
    num_channels = signals.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 10))
    for i in range(num_channels):
        axes[i].plot(signals[:, i])
    
    if show: plt.show()
    return fig
