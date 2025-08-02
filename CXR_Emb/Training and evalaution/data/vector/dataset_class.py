from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class VecDataset(Dataset):
    def __init__(self, info_df_path, data_path, labels=None):
        if labels is  None:
            self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Lesion',
            'Lung Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]
        else:
            self.PRED_LABEL = labels    
            


        self.info_df = pd.read_csv(info_df_path)
        self.labels = self.info_df[self.PRED_LABEL]
        self.sample_paths = self.info_df['Path']
        self.data_path = data_path

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        path = self.sample_paths.iloc[idx]
        # Handle any necessary filename formatting here
        if not path.endswith('.npy'):
            path = path + ".npy"

        full_item_path = self.data_path + path
        item = np.load(full_item_path)
        item = torch.from_numpy(item)

        label = self.labels.iloc[idx].values.astype(np.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return {'data': item, 'labels': label}
    