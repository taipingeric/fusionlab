from typing import *
import os
from torchvision.datasets.utils import download_and_extract_archive, download_url
import torch
from torch.utils.data import Dataset
from glob import glob
import scipy
import pandas as pd
from tqdm.auto import tqdm

URL_ECG = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
# URL_LABEL = "https://physionet.org/content/challenge-2017/1.0.0/REFERENCE-v3.csv"
URL_LABEL = "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv"

class ECGCSVClassificationDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 label_filename="REFERENCE-v3.csv",
                 target_leads=["lead"],
                 class_names=["N", "O", "A", "~"]):
        """
        Args:
            data_root (str): root directory of the dataset
            label_filename (str): filename of the label file
            target_leads (list): list of target lead names
            class_names (list): list of class names for mapping class name to class id
        """
        self.df_label = pd.read_csv(os.path.join(data_root, label_filename), header=None, names=["pat", "label"])
        self.signal_dir = os.path.join(data_root, "csv")
        self.signal_paths = sorted(glob(os.path.join(self.signal_dir, "*.csv")))
        class_names = ["N", "O", "A", "~"]
        self.class_map = {n: i for i, n in enumerate(class_names)}
        self.target_leads = target_leads
        print("dataset class map: ", self.class_map)

    def __len__(self):
        return len(self.signal_paths)
    
    def __getitem__(self, idx):
        row = self.df_label.iloc[idx]
        signal_filename = row["pat"] + ".csv"
        signal_path = os.path.join(self.signal_dir, signal_filename)
        df_csv = pd.read_csv(signal_path)
        signal = df_csv["lead"].values

        class_name = row["label"]
        class_id = self.class_map[class_name]

        signal = torch.tensor(signal, dtype=torch.float)
        class_id = torch.tensor(class_id, dtype=torch.long)

        # preprocess
        signal = signal.unsqueeze(0)
        return signal, class_id

    def _check_validate(self):
        assert len(self.df_label) == len(self.signal_paths), "csv files and label files are not matched"


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
    if not extract:
        download_root = os.path.expanduser(download_root)
        if not filename:
            filename = os.path.basename(url)

        download_url(url, download_root, filename, md5=None)
    else:
        if extract_root is None:
            extract_root = download_root
        download_and_extract_archive(url, download_root, extract_root, filename=filename)


def convert_mat_to_csv(root, target_dir="csv"):
    paths = glob(os.path.join(root, "training2017", "*.mat"))
    os.makedirs(os.path.join(root, target_dir), exist_ok=True)
    print("mat files: ", len(paths))
    print("start to convert mat files to csv files")
    for path in tqdm(paths):
        filename = os.path.basename(path)
        file_id = filename.split(".")[0]
        target_filename = file_id + ".csv"
        signal = scipy.io.loadmat(path)["val"][0]
        df = pd.DataFrame(columns=["lead"])
        df["lead"] = signal
        df.to_csv(os.path.join(root, target_dir, target_filename))

def validate_data(csv_dir, label_path):
    """
    check if the number of csv files and label files are matched
    """
    csv_paths = glob(os.path.join(csv_dir, "*.csv"))
    df_label = pd.read_csv(label_path, header=None, names=["pat", "label"])
    print("csv files: ", len(csv_paths))
    print("label files: ", len(df_label))
    assert len(csv_paths) == len(df_label), "csv files and label files are not matched"
    return
    

if __name__ == "__main__":
    root = "data"
    try:
        validate_data("./data/csv", "./data/REFERENCE-v3.csv")
    except:
        print("validation failed, start to donwload and convert data")
        download_file(URL_ECG, root, extract=True)
        download_file(URL_LABEL, root, extract=False)
        convert_mat_to_csv(root)

    ds = ECGCSVClassificationDataset("./data", label_filename="REFERENCE-v3.csv")

    sig, label = ds[0]
    print(sig.shape, label)    

    



