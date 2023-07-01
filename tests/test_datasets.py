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
            assert data_dict.keys() == set(['x', 'labels'])
            assert list(data_dict['x'].shape) == [NUM_FEATURES]
            pass

class TestLSTimeSegDataset:
    def test_LSTimeSegDataset(self, tmpdir):
        import numpy as np
        import os
        import pandas as pd
        import json
        import torch
        filename = "29be6360-12lead.csv"
        annotaion_path = os.path.join(tmpdir, "12.json")
        annotation = [
            {
                "csv": f"/data/upload/12/{filename}",
                "label": [
                {
                    "start": 0.004,
                    "end": 0.764,
                    "instant": False,
                    "timeserieslabels": ["N"]
                },
                {
                    "start": 0.762,
                    "end": 1.468,
                    "instant": False,
                    "timeserieslabels": ["p"]
                },
                {
                    "start": 1.466,
                    "end": 2.5,
                    "instant": False,
                    "timeserieslabels": ["t"]
                }
                ],
                "number": [{"number": 500}],
            }
        ]
        with open(annotaion_path, "w") as f:
            json.dump(annotation, f)
        col_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        df = pd.DataFrame()
        sample_rate = 500
        num_samples = sample_rate * 10
        df['time'] = np.arange(num_samples) / sample_rate
        for col_name in col_names:
            df[col_name] = np.random.randn(num_samples)
        df.to_csv(os.path.join(tmpdir, filename), index=False)
        ds = datasets.LSTimeSegDataset(data_dir=tmpdir,
                                       annotation_path=annotaion_path,
                                       class_map={"N": 1, "p": 2, "t": 3},
                                       column_names=col_names)
        signals, mask = ds[0]
        assert signals.shape == (len(col_names), num_samples)
        assert mask.shape == (num_samples, )
        assert type(signals) == torch.Tensor
        assert type(mask) == torch.Tensor
        assert len(ds) == 1