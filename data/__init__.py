#数据以及数据加载
import torch
import os
import pickle
import numpy as np
from .cifar10 import CIFAR10
from torchvision import transforms
def create_dataset(parser):
    loader = DataLoader(parser)
    return loader.getloader()

def Testdata(parser):
    fileroot  = parser.root
    trainkey = 'test'
    imgs = []
    labels = []
    for root, _, fnames in sorted(os.walk(fileroot)):
        for fname in fnames:
            if len(fnames) > 0 and trainkey in fname:
                path = os.path.join(root, fname)
                with open(path, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                    labels += dict['labels'.encode('utf8')]
                    for i in range(len(dict['data'.encode('utf8')])):
                        img = np.rot90(dict['data'.encode('utf8')][i].reshape((32, 32, 3), order='F'), -1)
                        imgs.append(img)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    cifar = CIFAR10(imgs, labels, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        cifar,
        batch_size=parser.batch_size,
        shuffle=parser.datashuffle,
        num_workers=int(parser.num_threads))
    return dataloader

class DataLoader(object):
    def __init__(self, parser):
        self.dataset = init_dataset(parser)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=parser.batch_size,
            shuffle= parser.datashuffle,
            num_workers=int(parser.num_threads))
    def getloader(self):
        return self.dataloader

def init_dataset(parser):
    fileroot  = parser.root
    trainkey = 'data' if parser.isTrain else 'test'
    imgs = []
    labels = []
    for root, _, fnames in sorted(os.walk(fileroot)):
        for fname in fnames:
            if len(fnames) > 0 and trainkey in fname:
                path = os.path.join(root, fname)
                with open(path, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                    labels += dict['labels'.encode('utf8')]
                    for i in range(len(dict['data'.encode('utf8')])):
                        img = np.rot90(dict['data'.encode('utf8')][i].reshape((32, 32, 3), order='F'), -1)
                        imgs.append(img)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    cifar = CIFAR10(imgs, labels, transform=transform)
    return cifar
