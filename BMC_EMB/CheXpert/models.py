import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import torch.nn.functional as F
import torch.nn as nn

# Define the MLP model


def create_mlp(input_dim, n_class):
    model = Sequential()
    model.add(Dense(768, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    if n_class == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(n_class, activation='softmax'))
    return model
