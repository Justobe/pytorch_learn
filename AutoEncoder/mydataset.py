from __future__ import print_function
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader


class MyDataset(data.Dataset):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __getitem__(self, index):  # 返回的是tensor
        input, target = self.words[index], self.labels[index]
        return input, target

    def __len__(self):
        return len(self.words)
