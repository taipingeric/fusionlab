from typing import *  # Importing all types from typing module
import os  # Importing os module
from glob import glob  # Importing glob function from glob module
from tqdm.auto import tqdm
from fusionlab.datasets.utils import download_file  # Importing tqdm function from tqdm.auto module

from scipy import io  # Importing scipy module
import pandas as pd  # Importing pandas module and aliasing it as pd

import torch  # Importing torch module
from torch.utils.data import Dataset  # Importing Dataset class from torch.utils.data module

URL_ECG = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
# URL_LABEL = "https://physionet.org/content/challenge-2017/1.0.0/REFERENCE-v3.csv"
URL_LABEL = "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv"


class ECGCSVClassificationDataset(Dataset):

    def __init__(self,
                 data_root,
                 label_filename="REFERENCE-v3.csv",
                 channels=["lead"],
                 class_names=["N", "O", "A", "~"]):
        """
        Args:
            data_root (str): root directory of the dataset
            label_filename (str): filename of the label file
            channels (list): list of target lead names
            class_names (list): list of class names for mapping class name to class id
        """
        # read the label file and store it in a pandas dataframe
        self.df_label = pd.read_csv(os.path.join(data_root, label_filename),
                                    header=None,
                                    names=["pat", "label"])
        # set the directory where the signal files are stored
        self.signal_dir = os.path.join(data_root, "csv")
        # get the paths of all the signal files and sort them
        self.signal_paths = sorted(glob(os.path.join(self.signal_dir, "*.csv")))
        # create a dictionary to map class names to class ids
        self.class_map = {n: i for i, n in enumerate(class_names)}
        # set the target leads for the dataset
        self.channels = channels
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
        assert len(self.df_label) == len(
            self.signal_paths), "csv files and label files are not matched"


def convert_mat_to_csv(root, target_dir="csv"):
    paths = glob(os.path.join(root, "training2017",
                              "*.mat"))  # get all paths of .mat files in the training2017 folder
    os.makedirs(os.path.join(root, target_dir),
                exist_ok=True)  # create a new directory named target_dir in the root directory
    print("mat files: ", len(paths))  # print the number of .mat files found
    print("start to convert mat files to csv files"
          )  # print a message indicating the start of the conversion process
    for path in tqdm(paths):  # iterate through each path in paths
        filename = os.path.basename(path)  # get the filename from the path
        file_id = filename.split(".")[
            0]  # get the file ID by splitting the filename at the "." and taking the first part
        target_filename = file_id + ".csv"  # create the target filename by appending ".csv" to the file ID
        signal = io.loadmat(path)["val"][0]  # load the .mat file and extract the "val" array
        df = pd.DataFrame(columns=["lead"
                                   ])  # create a new DataFrame with a single column named "lead"
        df["lead"] = signal  # set the "lead" column to the "val" array
        df.to_csv(
            os.path.join(root, target_dir, target_filename)
        )  # save the DataFrame as a CSV file in the target directory with the target filename


def validate_data(csv_dir, label_path):
    """
    check if the number of csv files and label files are matched
    """
    csv_paths = glob(os.path.join(csv_dir, "*.csv"))  # get all csv files in the directory
    df_label = pd.read_csv(label_path, header=None,
                           names=["pat", "label"])  # read the label file as a dataframe
    print("csv files: ", len(csv_paths))  # print the number of csv files
    print("label files: ", len(df_label))  # print the number of label files
    assert len(csv_paths) == len(
        df_label
    ), "csv files and label files are not matched"  # check if the number of csv files and label files are equal
    return  # return nothing


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
