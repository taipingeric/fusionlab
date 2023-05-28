'''
Lobachevsky University Electrocardiography Database (LUDB)
link: https://physionet.org/content/ludb/1.0.1/
patient id: 1~200
LEADS: i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6
segment annotaion symbols: p(P), N(QRS), t(T)
file extensions: .dat, .hea, .i, .ii, .iii, .avr, .avl, .avf, .v1, .v2, .v3, .v4, .v5, .v6
filename format: {patient_id}.{file extensions}

signal shape: (5000, 12)

annotation format:
symbols: ['(', 'N', ')', '(', 't', ')', '(', 'p', ')']
symbol tpye: ['p' 'N' 't' '(' ')']
sample: [ 641  664  690  773  840  887]
first symbol in signal index: 641 (all 200 patients)
last symbol in signal index: 3996 (all 200 patients)

ref: https://github.com/byschii/ecg-segmentation/blob/main/unet_for_ecg.ipynb

NOTE: since the annotation is channel independent, we only use the first channel (I) annotation
TODO: provide different annotation for different leads
'''
import os
from fusionlab.datasets.utils import download_file
from glob import glob
import numpy as np
import wfdb


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
    annotation = wfdb.rdann(DATA_FOLDER + "/" + str(index), extension=lead_name)
    return annotation


if __name__ == "__main__":
    # # download, extract and validate files
    # download_file(DATASET_URL, "./data", "./data/", "LUDB.zip", extract=True)
    # validate_files(f"./data/{DIR_NAME}")

    # extract symbols' index range
    for i in range(1, 201):
        lead_symbol = []
        lead_name = LEAD_NAMES[0] # take lead I
        annotation = get_annotation(f"./data/{DIR_NAME}/data", i, lead_name)
        symbol_list = annotation.symbol
        sample_list = annotation.sample
        # get all symbols '(' index in symbol list
        start_symbol_idxs = [i for i, s in enumerate(symbol_list) if s == '(']
        start_segment_idxs = [sample_list[i] for i in start_symbol_idxs]

        end_symbol_idxs = [i for i, s in enumerate(symbol_list) if s == ')']
        end_segment_idxs = [sample_list[i] for i in end_symbol_idxs]

        mid_symbol_idxs = [i for i, s in enumerate(symbol_list) if s != '(' and s != ')']
        mid_segment_idxs = [sample_list[i] for i in mid_symbol_idxs]

        for start, end, mid_symbol_idx in zip(start_segment_idxs, end_segment_idxs, mid_symbol_idxs):
            print(start, end, end-start, symbol_list[mid_symbol_idx])

        # print(start_idxs)


        # print(set(lead_symbol))
        break


    # for i in range(1, 201):
    #     signals = get_signal(f"./data/{DIR_NAME}/data", i)
    #     sig_len = signals.shape[0]
    #     print(sig_len)

    # record = wfdb.rdrecord(f"./data/{DIR_NAME}/data/1")
    # print(record)
    # # wfdb.plot_wfdb(record=record, title="Record 1 from LUDB", figsize=(10, 5))
    # # print(record.__dict__)
    # signals = record.p_signal  # (5000, 12)
    # sig_len = signals.shape[0]
    # print(signals.shape)

    # # save signal to csv
    # import pandas as pd
    # os.makedirs('./data/csv', exist_ok=True)
    # df = pd.DataFrame()
    # for i, lead_name in enumerate(LEAD_NAMES):
    #     df[lead_name] = signals[:, i]
    # df['time'] = np.arange(sig_len)
    # df.to_csv("./data/csv/1.csv", index=False)

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
