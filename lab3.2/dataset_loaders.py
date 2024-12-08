import torch
import re
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pandas as pd

class gsm_dataset(Dataset):
    def __init__(self, file_path):
        parquet_file = pq.read_table(file_path)
        answer=[]
        for i in range(len(parquet_file[1])):
            # if contains '#### ', then split it, remove all the , in the string
            answer.append(str(parquet_file[1][i]).split('#### ')[-1].replace(',', ''))
        self.data={
            'input': [str(i) for i in parquet_file[0]],
            'label': [str(i) for i in parquet_file[1]],
            'answer': [int(i) for i in answer]  
        }
    
    def __len__(self):
        return len(self.data['input'])
    
    def __getitem__(self, idx):
        item={
            'input': self.data['input'][idx],
            'label': self.data['label'][idx],
            'answer': self.data['answer'][idx]
        }
        return item