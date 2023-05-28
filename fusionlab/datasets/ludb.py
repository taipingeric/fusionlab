import os
from fusionlab.datasets.utils import download_file  # Importing tqdm function from tqdm.auto module
from glob import glob
import numpy as np

# ref: https://github.com/byschii/ecg-segmentation/blob/main/unet_for_ecg.ipynb

DATASET_URL = "https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip"
DIR_NAME = "lobachevsky-university-electrocardiography-database-1.0.1"
LEAD_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
CLS_NAMES = ['p', 'N', 't']
CLS_MAP = {n: i for i, n in enumerate(CLS_NAMES)}


def validate_files(data_dir):
    assert os.path.exists(data_dir)
    paths = glob(os.path.join(data_dir, "data", "*"))
    print(len(paths))
    extension_count = {}
    for p in paths:
        # get file type
        extension = os.path.basename(p).split(".")[1]
        extension_count[extension] = extension_count.get(extension, 0) + 1
    print(extension_count)


def get_signal(DATA_FOLDER, index: int):
    record = wfdb.rdrecord(DATA_FOLDER + "/" + str(index))
    assert type(record) is wfdb.Record
    return record.p_signal


# get annotations given the ecg lead
def get_annotation(DATA_FOLDER, index: int, lead_name: str):
    annotations = wfdb.rdann(DATA_FOLDER + "/" + str(index), extension=lead_name)
    print(annotations)
    print(annotations.symbol)
    print(set(annotations.symbol))
    print(len(annotations.symbol))
    print(np.array(annotations.sample))
    print(len(np.array(annotations.sample)))
    return annotations


if __name__ == "__main__":
    # download_file(DATASET_URL, "./data", "./data/", "LUDB.zip", extract=True)
    # validate_files(f"./data/{DIR_NAME}")
    import wfdb
    record = wfdb.rdrecord(f"./data/{DIR_NAME}/data/1")
    print(record)
    # wfdb.plot_wfdb(record=record, title="Record 1 from LUDB", figsize=(10, 5))
    # print(record.__dict__)
    signals = record.p_signal  # (5000, 12)
    sig_len = signals.shape[0]
    print(signals.shape)

    # save signal to csv
    import os
    import pandas as pd
    os.makedirs('./data/csv', exist_ok=True)
    df = pd.DataFrame()
    for i, lead_name in enumerate(LEAD_NAMES):
        df[lead_name] = signals[:, i]
    df['time'] = np.arange(sig_len)
    df.to_csv(f"./data/csv/1.csv", index=False)

    # # read ludb dataset annotations
    # # labels = wfdb.rdann(f"./data/{DIR_NAME}/data/1", "i")
    # labels = get_annotation(f"./data/{DIR_NAME}/data", 1, "i")

    # # cls_map = {
    # #     ')': 0,
    # #     't': 1,
    # #     'p': 2,
    # #     'N': 3,
    # #     '(': 4,
    # # }
    # SEGMENT_TO_COLOR = {
    #     'p': 'red', # p
    #     'N': 'blue', # qrs
    #     't': 'green', # t
    # }
    # # color with respect to the class index
    # color_map = np.random.rand(5, 3)

    # label_seq = np.zeros(sig_len, dtype=np.int64)

    # for i in range(len(labels.symbol)):
    #     start = labels.sample[i]
    #     end = labels.sample[i+1] if i+1 < len(labels.symbol) else sig_len
    #     label_seq[start:end] = cls_map[labels.symbol[i]]

    # print(label_seq)

    # import matplotlib.pyplot as plt
    # # plot the signal with color map

    # plt.plot(signals[:, 0])
    # # vertical area with color
    # for i in range(len(labels.symbol)):
    #     start = labels.sample[i]
    #     end = labels.sample[i+1] if i+1 < len(labels.symbol) else sig_len
    #     plt.axvspan(start, end, color=color_map[cls_map[labels.symbol[i]]], alpha=0.5)
    # # plt.show()

    # # plt.plot(signals[:, 0])
    # plt.plot(label_seq)
    # plt.show()
