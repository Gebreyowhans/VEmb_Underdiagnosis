# import numpy as np
from torchsummary import summary
from torchvision import models
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.nn as nn
import pickle
import pandas as pd
import torch
# from classification.train import train
# from classification.predictions import make_pred_multilabel
# from LearningCurve import PlotLearnignCurve
import sys
# ----------------------------- q
# "/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"#
#  /datasets/mimic-cxr/physionet.org/files/mimic-cxr/2.0.0/files

# CheXpert-v1.0-small/train/patient49643/study3/view1_frontal.jpg,

PATH_TO_IMAGES = '/datasets/chexpert/'

df_path = "/h/gebrehb/gebrehb_link/ClipEmbedding/dataframes/chexpert_df.csv"


def main():
    df = pd.read_csv(df_path)


if __name__ == "__main__":
    main()
    #
