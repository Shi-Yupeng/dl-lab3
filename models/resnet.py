# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

    def forward(self, x):
        return 0