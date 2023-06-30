from fusionlab import datasets

class TestHFDataset:
    def test_HFDataset(self):
        import torch
        print("Test HFDataset")

        NUM_DATA = 20
        NUM_FEATURES = 16
        ds = torch.utils.data.TensorDataset(
            torch.zeros(NUM_DATA, NUM_FEATURES),
            torch.zeros(NUM_DATA))
        for x, y in ds:
            assert list(x.shape) == [NUM_FEATURES]
            pass
        hf_ds = datasets.HFDataset(ds)
        assert hf_ds is not None
        for i in range(len(hf_ds)):
            data_dict = hf_ds[i]
            assert data_dict.keys() == set('x', 'labels')
            assert list(data_dict['x'].shape) == [NUM_FEATURES]
            pass
