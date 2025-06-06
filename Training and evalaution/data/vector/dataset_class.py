from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import numpy as np


class VecDataset_All(Dataset):
    def __init__(self, info_df_path_mimic, info_df_path_chexpert, data_path_mimic, data_path_chexpert):
        self.PRED_LABEL= [
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
            'Support Devices']


        self.info_df_mimic = pd.read_csv(info_df_path_mimic)
        self.info_df_chexpert = pd.read_csv(info_df_path_chexpert)
        self.lables_mimic = self.info_df_mimic[self.PRED_LABEL]
        self.lables_chexpert = self.info_df_chexpert[self.PRED_LABEL]
        self.samples_path_mimic = self.info_df_mimic['path']
        self.samples_path_chexpert = self.info_df_chexpert['path']
        self.data_path_mimic = data_path_mimic
        self.data_path_chexpert = data_path_chexpert
        self.len_mimic = len(self.info_df_mimic)
        self.len_chexpert = len(self.info_df_chexpert)

    def __len__(self):

        return self.len_mimic + self.len_chexpert
    
    def __getitem__(self, idx):

        if idx < self.len_mimic:
            
            path = self.samples_path_mimic.iloc[idx]
            path = path + ".npy"
            full_item_path = self.data_path_mimic + path
            item = np.load(full_item_path)
            #convert to tensor
            item = torch.from_numpy(item)
            # read labels from labels_df
            label = self.lables_mimic.iloc[idx]
            label = list(label)
            label = torch.tensor(label, dtype=torch.float32)

       
            return {'data':item, 'labels': label}
        else:
            
            idx = idx - len(self.info_df_mimic)
            path = self.samples_path_chexpert.iloc[idx]
            path = path.replace("/", "_").replace(".", "_") + ".npy"
            # first char of path to / instead of _
            path = "/" + path[1:]
            full_item_path = self.data_path_chexpert + path
            item = np.load(full_item_path)
            #convert to tensor
            item = torch.from_numpy(item)
            # read labels from labels_df
            label = self.lables_chexpert.iloc[idx]
            label = list(label)
            label = torch.tensor(label, dtype=torch.float32)
    
            return {'data':item, 'labels': label}
