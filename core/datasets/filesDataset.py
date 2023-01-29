import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob

class FilesDataset(Dataset):
    def __init__(self, files = None, path = None, transform = None):
        if files is not None:
            self.files = files
        else:
            self.files = glob.glob(path + '/*.npz')
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw_feature = np.load(self.files[idx])

        feature = {}

        for key, value in raw_feature.items():
            if key == 'scenario/id':
                feature[key] = str(value, encoding = "utf-8")
            else:
                feature[key] = torch.from_numpy(value)

        if self.transform is not None:
            feature = self.transform(feature)

        return feature


    