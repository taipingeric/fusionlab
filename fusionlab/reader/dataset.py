import pandas as pd
import torch.utils.data as tud  # 導入PyTorch的資料處理工具
from ai4ecg.reader import csvread  # 讀取ECG signal


# 做一個簡單的Dataset class
class ECGDataset(tud.Dataset):
    def __init__(self,
                annotation_file,  # 標註檔案路徑
                data_dir,  # ECG資料路徑
                transform=None):  # 資料轉換函式
        self.data_dir = data_dir
        self.ecg_signals = pd.read_csv(annotation_file)  # 讀取標註檔案
        self.transform = transform
    def __len__(self):
        return len(self.ecg_signals)  # 回傳資料筆數
    def __getitem__(self, idx):
        entry = self.ecg_signals.iloc[idx]  # 取得指定索引的資料
        data = csvread.read_csv(self.data_dir+entry['filename'])  # 讀取ECG資料
        label = entry['label']  # 取得標籤
        if self.transform:
            data = self.transform(data)  # 資料轉換
        return data, label  # 回傳資料和標籤