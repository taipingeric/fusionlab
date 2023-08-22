import torch
from torchvision.datasets.utils import download_and_extract_archive, download_url
import os
from typing import Optional
from glob import glob
import json
import pandas as pd
import numpy as np

def download_file(url: str,
                  download_root: str,
                  extract_root: Optional[str] = None,
                  filename: Optional[str] = None,
                  extract=False) -> None:
    """
    Download a file from a url and optionally extract it to a target directory.
    Args:
        url (str): URL to download file from
        download_root (str): Directory to place downloaded file in
        extract_root (str, optional): Directory to extract downloaded file to
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        extract (bool, optional): If True, extract the downloaded file. Otherwise, do not extract.
    """
    if not extract:  # if extract is False
        download_root = os.path.expanduser(download_root)  # expand the user's home directory in download_root
        if not filename:  # if filename is not provided
            filename = os.path.basename(url)  # set filename to the basename of the URL

        download_url(url, download_root, filename, md5=None)  # download the file
    else:
        if extract_root is None:  # if extract_root is not provided
            extract_root = download_root  # set extract_root to download_root
        download_and_extract_archive(url, download_root, extract_root, filename=filename)  # download and extract the file to extract_root

class HFDataset(torch.utils.data.Dataset):
    """
    Base Hugginface dataset wrapper class
    Args:
        dataset: a dataset object that contains a getitem method
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        x, labels = self.dataset[index]  # Forward pass the dataset
        return {'x': x, 'labels': labels}


class LSTimeSegDataset(torch.utils.data.Dataset):
    """
    Dataset for label-studio timeseries segmentation task
    """
    
    def __init__(self, data_dir, annotation_path, class_map, column_names):
        """
        Dataset for label-studio timeseries segmentation task
        
        Args:
            data_dir (str): directory of csv files
            annotation_path (str): path to annotation json file
            class_map (dict): a dictionary mapping class names to class indices
            column_names (List[str]): A list of column names for the signal data in the CSV files.
        
        Examples::
            >>> ds = LSTimeSegDataset(data_dir="./12",
            >>>                       annotation_path="./12.json",
            >>>                       class_map={"N": 1, "p": 2, "t": 3},
            >>>                       column_names=['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
            >>> signals, mask = ds[0]
        """
        super().__init__()
        self.data_dir = data_dir
        self.annotation_path = annotation_path
        self.class_map = class_map
        self.column_names = column_names
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
        Returns:
            signals (torch.Tensor): shape: (num_samples, Channels)
            mask (torch.Tensor): shape: (num_samples, )

        """
        annotation = self.annotations[index]
        signal_filename = annotation["csv"].split(os.sep)[-1]
        signal_path = os.path.join(self.data_dir, signal_filename)
        df = pd.read_csv(signal_path)
        mask = np.zeros(len(df), dtype=np.int32)
        for segment in annotation["label"]:
            start_time = segment["start"]
            end_time = segment["end"]
            class_name = segment["timeserieslabels"][0]
            start_idx = df[df["time"] == start_time].index[0]
            end_idx = df[df["time"] == end_time].index[0]
            mask[start_idx: end_idx] = self.class_map[class_name]
        signals = df[self.column_names].values

        signals = torch.from_numpy(signals).float().permute(1, 0)
        signals = self.preprocess(signals)
        mask = torch.from_numpy(mask).long()
        return signals, mask
    
    def preprocess(self, signals):
        return standardize_tensor(signals, dim=1)

class LSTimeClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for label-studio timeseries classification task
    """
    
    def __init__(self, data_dir, annotation_path, class_map, column_names):
        """
        Dataset for label-studio timeseries segmentation task
        
        Args:
            data_dir (str): directory of csv files
            annotation_path (str): path to annotation json file
            class_map (dict): a dictionary mapping class names to class indices
            column_names (List[str]): A list of column names for the signal data in the CSV files.
        
        Examples::
            >>> ds = LSTimeClassificationDataset(
            >>>     data_dir=DATA_DIR,
            >>>     annotation_path=ANNOTATION_PATH,
            >>>     class_map={"Normal": 1, "AF": 2, "AV Block": 3, "Noise": 4},
            >>>     column_names=['i', 'ii', 'iii'])
            >>> signals, label = ds[0]
        """
        super().__init__()
        self.data_dir = data_dir
        self.annotation_path = annotation_path
        self.class_map = class_map
        self.column_names = column_names
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
        Returns:
            signals (torch.Tensor): shape: (num_samples, Channels)
            mask (torch.Tensor): shape: (num_samples, )
            
        """
        annotation = self.annotations[index]
        signal_filename = annotation["csv"].split(os.sep)[-1]
        signal_path = os.path.join(self.data_dir, signal_filename)
        df = pd.read_csv(signal_path)
        signals = df[self.column_names].values
        # signals = self.preprocess(signals)
        label_name = annotation['pattern']
        label = self.class_map[label_name]

        signals = torch.from_numpy(signals).float().permute(1, 0) # (L, C) -> (C, L)
        signals = self.preprocess(signals)
        label = torch.tensor(label).long()
        return signals, label
    

    def preprocess(self, signals):
        return standardize_tensor(signals, dim=1)


def standardize_tensor(tensor, dim=0):
    """
    Standardize a tensor by channel

    Args:
        tensor (torch.Tensor): shape: (Channels, num_samples)
        dim (int): dimension to standardize
    Returns:
        tensor (torch.Tensor): shape: (Channels, num_samples,)
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    tensor = (tensor - mean) / std
    return tensor


# TODO: add test
def count_parameters(
        model: torch.nn.Module, 
        trainable_only: bool = False
    ) -> int:
    """
    Returns the number of parameters in a model

    Args:
        model: a pytorch model
        trainable_only: if True, only count trainable parameters

    Returns:
        num_parameters: number of parameters in the model

    Reference: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    import torch.nn as nn
    layer = nn.Linear(10, 10)
    
    for p in layer.parameters():
        p.requires_grad = False
    print(count_parameters(layer, trainable_only=False))
    print(count_parameters(layer, trainable_only=True))