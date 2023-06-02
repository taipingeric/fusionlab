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

The generated annotation file follows the format of the label-studio timeserieslabels segment annotation

NOTE: since the annotation is channel independent, we only use the first channel (I) annotation
TODO: provide different annotation for different leads
'''
import os
from fusionlab.datasets.utils import download_file
from glob import glob
import numpy as np
import wfdb
from tqdm.auto import tqdm
import json
import torch


DATASET_URL = "https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip"
DIR_NAME = "lobachevsky-university-electrocardiography-database-1.0.1"
TARGET_FILENAME = "LUDB.zip"
LEAD_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
CLS_NAMES = ['p', 'N', 't']
CLS_MAP = {n: i+1 for i, n in enumerate(CLS_NAMES)}
PAT_ID_LIST = list(range(1, 201))
CLS_COLOR = {
    1: 'red', # p
    2: 'blue', # qrs
    3: 'green', # t
}

# plot signal with annotation
def plot(signal, label_seq):
    import matplotlib.pyplot as plt
    plt.plot(signal[:, 0].numpy())
    # fill range with matplotlib axvspan
    for i in range(len(label_seq)):
        if label_seq[i] != 0:
            plt.axvspan(i, i+1, color=CLS_COLOR[label_seq[i].item()], alpha=0.5, 
                        linewidth=0,
                        )
    plt.show()

class LUDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotation_path, transform=None, start_idx=641, end_idx=3996):
        # validate raw files
        try:
            self.validate_files(data_dir)
        except Exception as e:
            print(e)
            print(f"Start downloading the dataset from {DATASET_URL} to {os.path.join(data_dir, TARGET_FILENAME)}")
            download_file(DATASET_URL, data_dir, data_dir, TARGET_FILENAME, extract=True)

        # validate annotation file
        if not os.path.exists(annotation_path):
            print(f"Start processing annotation file and save to {annotation_path}")
            self.process_annotation(annotation_path)

        self.data_dir = data_dir
        self.signal_dir = os.path.join(data_dir, DIR_NAME, "data")
        self.annotation_path = annotation_path
        with open(self.annotation_path, "r") as f:
            self.annotations = json.load(f)
        self.file_ids = [int(anno['csv'].split(os.sep)[-1].split('.')[0]) for anno in self.annotations]
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.transform = transform
        self.data = []
        self.labels = []

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        anno = self.annotations[index]
        signal = self.get_signal(self.signal_dir, file_id) # (5000, 12)
        signal_len = signal.shape[0]
        label_seq = self.map_annotaion_to_label_seq(anno, signal_len)
        signal, label_seq = self.extract_signal_label(signal, label_seq)

        if self.transform:
            signal = self.transform(signal)

        signal = torch.tensor(signal, dtype=torch.float)
        label_seq = torch.tensor(label_seq, dtype=torch.long)
        return signal, label_seq

    def __len__(self):
        return len(self.file_ids)
    
    def extract_signal_label(self, signal, label):
        # extract signal and label with respect to start and end index
        return signal[self.start_idx:self.end_idx, :], label[self.start_idx:self.end_idx]
    
    # validate number of files and file types
    def validate_files(self, data_dir):
        assert os.path.exists(data_dir)
        paths = glob(os.path.join(data_dir, DIR_NAME, "data", "*"))
        if len(paths) == 0:
            raise Exception(f"No files found in the dataset folder: {data_dir}")
        
        extension_count = {}
        for p in paths:
            # get file type
            extension = os.path.basename(p).split(".")[1]
            extension_count[extension] = extension_count.get(extension, 0) + 1
        keys = list(extension_count.keys())
        counts = list(extension_count.values())
        if sorted(keys) != sorted(['dat', 'hea'] + LEAD_NAMES):
            raise Exception(f"Invalid file types: {keys}")
        if len(set(counts)) > 1:
            raise Exception(f"Different number of files for each file type: {extension_count}")
    
    # process annotation file and save to json
    def process_annotation(self, export_path):
        label_list = []
        # extract symbols' index range
        for pat_id in tqdm(PAT_ID_LIST):
            label_dict = self.get_segment_annotation(pat_id)
            label_list.append(label_dict)

        # save label list to json
        with open(export_path, "w") as f:
            json.dump(label_list, f)

    # map json annotation to label sequence
    def map_annotaion_to_label_seq(self, annotation, sig_len):
        label_seq = np.zeros(sig_len, dtype=int)
        for segment in annotation["label"]:
            start = segment["start"]
            end = segment["end"]
            segment_class = segment["timeserieslabels"][0]
            segment_class_idx = CLS_MAP[segment_class]
            label_seq[start:end] = segment_class_idx
        return label_seq
    
    def get_signal(self, DATA_FOLDER, index: int):
        record = wfdb.rdrecord(DATA_FOLDER + "/" + str(index))
        assert type(record) is wfdb.Record
        return record.p_signal


    # get annotations given the ecg lead
    def get_annotation(self, DATA_FOLDER, index: int, lead_name: str):
        annotation = wfdb.rdann(DATA_FOLDER + "/" + str(index), extension=lead_name)
        return annotation

    def get_segment_annotation(self, pat_id):
        # init label dict
        label_dict = {
            "csv": f"{pat_id}.csv",
            "label": []
        }
        lead_name = LEAD_NAMES[0] # take lead I
        annotation = self.get_annotation(f"./data/{DIR_NAME}/data", pat_id, lead_name)
        symbol_list = annotation.symbol
        sample_list = annotation.sample
        # get all start symbols '(' index in symbol list
        start_symbol_idxs = [i for i, s in enumerate(symbol_list) if s == '(']
        start_segment_idxs = [sample_list[i] for i in start_symbol_idxs]
        # get all start symbols ')' index in symbol list
        end_symbol_idxs = [i for i, s in enumerate(symbol_list) if s == ')']
        end_segment_idxs = [sample_list[i] for i in end_symbol_idxs]
        # get all mid symbols index in symbol list
        mid_symbol_idxs = [i for i, s in enumerate(symbol_list) if s != '(' and s != ')']

        for start, end, mid_symbol_idx in zip(start_segment_idxs, end_segment_idxs, mid_symbol_idxs):
            symbol = symbol_list[mid_symbol_idx]
            segment_dict = {
                "start": int(start),
                "end": int(end),
                "timeserieslabels": [symbol]
            }
            label_dict["label"].append(segment_dict)
        return label_dict

if __name__ == "__main__":
    ds = LUDBDataset("./data", "./data/ludb_annotation.json")
    signal, label_seq = ds[0]
    plot(signal, label_seq)
