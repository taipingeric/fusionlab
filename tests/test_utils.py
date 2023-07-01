class TestPlotChannels:
    def test_plot_channels(self):
        from fusionlab.utils import plot_channels
        import numpy as np
        import matplotlib
        """Test that plot_channels() returns a figure."""
        signals = np.random.randn(500, 12)
        fig = plot_channels(signals, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)