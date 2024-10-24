import os
import h5py
import time
import torch
import pickle
import logging
import numpy as np

from torch.utils.data import Dataset
from class_dataloader.utils import window_select

class Train(Dataset):
    def __init__(self, configs):
        self.path = configs['train_path']
        self.window = configs['window']
        self.log = pickle.load(open(self.path + 'log.p', "rb"))

    def __len__(self):
        return self.log[-1]

    def __getitem__(self, idx):
        f = np.where(self.log<=idx)[0][-1]
        local_path = os.path.join(self.path, str(self.log[f]))
        tactile, heatmap, keypoint = window_select(self.log, local_path, f, idx, self.window)
        return idx, tactile, heatmap, keypoint