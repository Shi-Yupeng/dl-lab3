# -*- coding:utf-8 -*-
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self, datas, labels, transform=None):
        super(CIFAR10, self).__init__()
        self.samples = []
        for data, label in zip(datas, labels):
            self.samples.append((data, label))
        self.transform = transform

    def __getitem__(self, item):
        data, label = self.samples[item]
        if self.transform is not None:
            data = self.transform(data.copy())
        return data, label

    def __len__(self):
        return len(self.samples)