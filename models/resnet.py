# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        shortcut = self.shortcut(out)
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, BasicBlock, num_classes=10):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, num_blocks, stride):

    def forward(self, x):
        return 0