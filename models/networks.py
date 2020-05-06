import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet

#class
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
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
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
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(SEBasicBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()
        self.se = SEBlock(out_channel)

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel),
            )


    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.se(out)
        out += shortcut
        return out


# 用于构建ResNet50， ResNet101， ResNet152 由于模型过大训练不一定可行，先不实现
# class Bottleneck(nn.Module):
#     def __init__(self, in_channel, out_channel, stride=1):
#         super(Bottleneck, self).__init__()
#
#     def forward(self, x):
#         return 0


class SEResNet(nn.Module):
    def __init__(self, block=SEBasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(SEResNet, self).__init__()
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



# VGG19网络
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.feature = nn.Sequential()
        self.feature.add_module('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
        self.feature.add_module('bn1_1', nn.BatchNorm2d(64))
        self.feature.add_module('relu1_1', nn.ReLU(inplace=True))

        self.feature.add_module('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.feature.add_module('bn1_2', nn.BatchNorm2d(64))
        self.feature.add_module('relu1_2', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature.add_module('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.feature.add_module('bn2_1', nn.BatchNorm2d(128))
        self.feature.add_module('relu2_1', nn.ReLU(inplace=True))

        self.feature.add_module('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.feature.add_module('bn2_2', nn.BatchNorm2d(128))
        self.feature.add_module('relu2_2', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))


        self.feature.add_module('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.feature.add_module('bn3_1', nn.BatchNorm2d(256))
        self.feature.add_module('relu3_1', nn.ReLU(inplace=True))

        self.feature.add_module('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature.add_module('bn3_2', nn.BatchNorm2d(256))
        self.feature.add_module('relu3_2', nn.ReLU(inplace=True))

        self.feature.add_module('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature.add_module('bn3_3', nn.BatchNorm2d(256))
        self.feature.add_module('relu3_3', nn.ReLU(inplace=True))

        self.feature.add_module('conv3_4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature.add_module('bn3_4', nn.BatchNorm2d(256))
        self.feature.add_module('relu3_4', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))


        self.feature.add_module('conv4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn4_1', nn.BatchNorm2d(512))
        self.feature.add_module('relu4_1', nn.ReLU(inplace=True))

        self.feature.add_module('conv4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn4_2', nn.BatchNorm2d(512))
        self.feature.add_module('relu4_2', nn.ReLU(inplace=True))

        self.feature.add_module('conv4_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn4_3', nn.BatchNorm2d(512))
        self.feature.add_module('relu4_3', nn.ReLU(inplace=True))

        self.feature.add_module('conv4_4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn4_4', nn.BatchNorm2d(512))
        self.feature.add_module('relu4_4', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))


        self.feature.add_module('conv5_1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn5_1', nn.BatchNorm2d(512))
        self.feature.add_module('relu5_1', nn.ReLU(inplace=True))

        self.feature.add_module('conv5_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn5_2', nn.BatchNorm2d(512))
        self.feature.add_module('relu5_2', nn.ReLU(inplace=True))

        self.feature.add_module('conv5_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn5_3', nn.BatchNorm2d(512))
        self.feature.add_module('relu5_3', nn.ReLU(inplace=True))

        self.feature.add_module('conv5_4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature.add_module('bn5_4', nn.BatchNorm2d(512))
        self.feature.add_module('relu5_4', nn.ReLU(inplace=True))
        self.feature.add_module('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))

        self.classify = nn.Sequential()
        self.classify.add_module('FC1', nn.Linear(512*1*1, 4096))
        self.classify.add_module('BN1', nn.BatchNorm1d(4096))
        self.classify.add_module('Relu1', nn.ReLU(inplace=True))
        self.classify.add_module('Dropout1', nn.Dropout(0.5))

        self.classify.add_module('FC2', nn.Linear(4096, 4096))
        self.classify.add_module('BN2', nn.BatchNorm1d(4096))
        self.classify.add_module('Relu2', nn.ReLU(inplace=True))
        self.classify.add_module('Dropout2', nn.Dropout(0.5))

        self.classify.add_module('FC3', nn.Linear(4096, 10))
        # self.classify.add_module('BN3', nn.BatchNorm1d(10))
        # self.classify.add_module('Relu3', nn.ReLU(inplace=True))
        # self.classify.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, x):
        x = self.feature(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classify(x)
        return x

# VGG19_SE网络
class SEVGG(nn.Module):
    def __init__(self):
        super(SEVGG, self).__init__()

        self.feature1 = nn.Sequential()
        self.feature1.add_module('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
        self.feature1.add_module('bn1_1', nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature1.add_module('relu1_1', nn.ReLU(inplace=True))
        self.se1 = SE_block(ch_in=64)

        self.feature2 = nn.Sequential()
        self.feature2.add_module('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.feature2.add_module('bn1_2', nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature2.add_module('relu1_2', nn.ReLU(inplace=True))
        self.se2 = SE_block(ch_in=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.feature3 = nn.Sequential()
        self.feature3.add_module('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.feature3.add_module('bn2_1', nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature3.add_module('relu2_1', nn.ReLU(inplace=True))
        self.se3 = SE_block(ch_in=128)

        self.feature4 = nn.Sequential()
        self.feature4.add_module('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.feature4.add_module('bn2_2', nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature4.add_module('relu2_2', nn.ReLU(inplace=True))
        self.se4 = SE_block(ch_in=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feature5 = nn.Sequential()
        self.feature5.add_module('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.feature5.add_module('bn3_1', nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature5.add_module('relu3_1', nn.ReLU(inplace=True))
        self.se5 = SE_block(ch_in=256)

        self.feature6 = nn.Sequential()
        self.feature6.add_module('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature6.add_module('bn3_2', nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature6.add_module('relu3_2', nn.ReLU(inplace=True))
        self.se6 = SE_block(ch_in=256)

        self.feature7 = nn.Sequential()
        self.feature7.add_module('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature7.add_module('bn3_3', nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature7.add_module('relu3_3', nn.ReLU(inplace=True))
        self.se7 = SE_block(ch_in=256)

        self.feature8 = nn.Sequential()
        self.feature8.add_module('conv3_4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.feature8.add_module('bn3_4', nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature8.add_module('relu3_4', nn.ReLU(inplace=True))
        self.se8 = SE_block(ch_in=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feature9 = nn.Sequential()
        self.feature9.add_module('conv4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
        self.feature9.add_module('bn4_1', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature9.add_module('relu4_1', nn.ReLU(inplace=True))
        self.se9 = SE_block(ch_in=512)

        self.feature10 = nn.Sequential()
        self.feature10.add_module('conv4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature10.add_module('bn4_2', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature10.add_module('relu4_2', nn.ReLU(inplace=True))
        self.se10 = SE_block(ch_in=512)

        self.feature11 = nn.Sequential()
        self.feature11.add_module('conv4_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature11.add_module('bn4_3', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature11.add_module('relu4_3', nn.ReLU(inplace=True))
        self.se11 = SE_block(ch_in=512)

        self.feature12 = nn.Sequential()
        self.feature12.add_module('conv4_4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature12.add_module('bn4_4', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature12.add_module('relu4_4', nn.ReLU(inplace=True))
        self.se12 = SE_block(ch_in=512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feature13 = nn.Sequential()
        self.feature13.add_module('conv5_1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature13.add_module('bn5_1', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature13.add_module('relu5_1', nn.ReLU(inplace=True))
        self.se13 = SE_block(ch_in=512)

        self.feature14 = nn.Sequential()
        self.feature14.add_module('conv5_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature14.add_module('bn5_2', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature14.add_module('relu5_2', nn.ReLU(inplace=True))
        self.se14 = SE_block(ch_in=512)

        self.feature15 = nn.Sequential()
        self.feature15.add_module('conv5_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature15.add_module('bn5_3', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature15.add_module('relu5_3', nn.ReLU(inplace=True))
        self.se15 = SE_block(ch_in=512)

        self.feature16 = nn.Sequential()
        self.feature16.add_module('conv5_4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.feature16.add_module('bn5_4', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.feature16.add_module('relu5_4', nn.ReLU(inplace=True))
        self.se16 = SE_block(ch_in=512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classify = nn.Sequential()
        self.classify.add_module('FC1', nn.Linear(512*1*1, 1024))
        # self.classify.add_module('BN1', nn.BatchNorm1d(1024))
        self.classify.add_module('Relu1', nn.ReLU(inplace=True))
        self.classify.add_module('Dropout1', nn.Dropout(0.5))

        self.classify.add_module('FC2', nn.Linear(1024, 1024))
        # self.classify.add_module('BN2', nn.BatchNorm1d(1024))
        self.classify.add_module('Relu2', nn.ReLU(inplace=True))
        self.classify.add_module('Dropout2', nn.Dropout(0.5))

        self.classify.add_module('FC3', nn.Linear(1024, 10))
        # self.classify.add_module('BN3', nn.BatchNorm1d(10))
        # self.classify.add_module('Relu3', nn.ReLU(inplace=True))
        # self.classify.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, x):
        x = self.feature1(x)
        s = self.se1(x)
        x = x * s.expand_as(x)

        x = self.feature2(x)
        x = self.maxpool1(x)
        s = self.se2(x)
        x = x * s.expand_as(x)

        x = self.feature3(x)
        s = self.se3(x)
        x = x * s.expand_as(x)

        x = self.feature4(x)
        x = self.maxpool2(x)
        s = self.se4(x)
        x = x * s.expand_as(x)

        x = self.feature5(x)
        s = self.se5(x)
        x = x * s.expand_as(x)

        x = self.feature6(x)
        s = self.se6(x)
        x = x * s.expand_as(x)

        x = self.feature7(x)
        s = self.se7(x)
        x = x * s.expand_as(x)

        x = self.feature8(x)
        x = self.maxpool3(x)
        s = self.se8(x)
        x = x * s.expand_as(x)

        x = self.feature9(x)
        s = self.se9(x)
        x = x * s.expand_as(x)

        x = self.feature10(x)
        s = self.se10(x)
        x = x * s.expand_as(x)

        x = self.feature11(x)
        s = self.se11(x)
        x = x * s.expand_as(x)

        x = self.feature12(x)
        x = self.maxpool4(x)
        s = self.se12(x)
        x = x * s.expand_as(x)

        x = self.feature13(x)
        s = self.se13(x)
        x = x * s.expand_as(x)

        x = self.feature14(x)
        s = self.se14(x)
        x = x * s.expand_as(x)

        x = self.feature15(x)
        s = self.se15(x)
        x = x * s.expand_as(x)

        x = self.feature16(x)
        x = self.maxpool5(x)
        s = self.se16(x)
        x = x * s.expand_as(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classify(x)
        return x

class SE_block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_block, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential()
        self.se_block.add_module('linear1', nn.Linear(ch_in, ch_in // reduction, bias=False))
        self.se_block.add_module('Relu', nn.ReLU(inplace=True))
        self.se_block.add_module('linear2', nn.Linear(ch_in // reduction, ch_in, bias=False))
        self.se_block.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        h, w, _, _ = x.size()
        avg = self.avg(x).view(h, w)
        s = self.se_block(avg).view(h, w, 1, 1)
        return s




#functions
def Criterion():
    return nn.CrossEntropyLoss()





def definenet(opt):
    class_str = opt.model
    CLASS = eval(class_str)()
    return CLASS
# a = definenet('ResNet34')
# print(a)