# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


# 用于构建ResNet50， ResNet101， ResNet152 由于模型过大训练不一定可行，先不实现
# class Bottleneck(nn.Module):
#     def __init__(self, in_channel, out_channel, stride=1):
#         super(Bottleneck, self).__init__()
#
#     def forward(self, x):
#         return 0


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1 * 1, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), 512 * 1 * 1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])
#
#
# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])
#
#
# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])
