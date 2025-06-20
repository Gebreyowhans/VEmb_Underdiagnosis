
import torch
from torch.utils.data import Dataset
import numpy as np
from imageio import imread
from PIL import Image
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, dataframe_path, path_image, finding="any", transform=None, labels=None):
        self.dataframe = pd.read_csv(dataframe_path)
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        # "/datasets/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.path_image = path_image

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.dataframe.columns:
                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")
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
                'Support Devices']
        else:
            self.PRED_LABEL = labels

    def __getitem__(self, idx):
    


        item = self.dataframe.iloc[idx]
        
        
        img = imread(self.path_image + item["Path"])
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        sample = {'data':img, 'labels': np.array(list(label))}

        return sample


    def __len__(self):
        return self.dataset_size




class ImageDatasetRaceOrGender(Dataset):
    """
    Dataset that returns (image, one-hot label) pairs for either race or gender.

    Args
    ----
    dataframe_path : str
        CSV file containing a `Path` column (with the relative image path)
        plus the column that stores race OR gender.
    path_image : str
        Root folder that, when concatenated with the value in dataframe['Path'],
        yields the full image filename.
    label_col : str
        Name of the column that holds the attribute (e.g. 'race' or 'gender').
    transform : torchvision transform, optional
        Usual transform pipeline.
    """
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, dataframe_path: str,
                 path_image: str,
                 label_col: str,
                 transform=None):
        super().__init__()

        self.df = pd.read_csv(dataframe_path)
        self.path_image = path_image.rstrip("/") + "/"
        self.label_col = label_col
        self.transform = transform

        # ――― Decide which mapping to use ―――
        col_low = label_col.lower()
        if "gender" in col_low or "sex" in col_low:
            # Male / Female → 2-way one-hot
            self.label_names = ["Male", "Female"]
            self.map_str_to_vec = {
                "male":  torch.tensor([1.0, 0.0]),
                "female": torch.tensor([0.0, 1.0]),
            }
            self.unknown = torch.tensor([0.0, 0.0])     # for blanks / misc

        # mapping depends to the expriment for example include the Other category or exclude it or only white vs black/african american
        elif "race" in col_low or "ethnic" in col_low:
            # 4-way one-hot: WHITE, BLACK/AFRICAN AMERICAN, ASIAN, OTHER
            self.label_names = [
                "WHITE",
                "BLACK/AFRICAN AMERICAN",
                "ASIAN",
                "OTHER",
            ]
            self.map_str_to_vec = {
                "white":  torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "black/african american": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "asian":  torch.tensor([0.0, 0.0, 1.0, 0.0]),
            }
            self.unknown = torch.tensor([0.0, 0.0, 0.0, 1.0])  # OTHER
        else:
            raise ValueError(
                f"Column '{label_col}' not recognised as race or gender."
            )

    # ──────────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    # ──────────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Image ----
        img = imread(self.path_image + row["Path"])
        if img.ndim == 2:                          # greyscale → 3-channel
            img = np.stack([img]*3, axis=2)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        # ---- Label ----
        raw_val = str(row[self.label_col]).strip().lower()
        label_vec = self.map_str_to_vec.get(raw_val, self.unknown).clone()

        return {
            "data": img,
            "labels": label_vec

        }

