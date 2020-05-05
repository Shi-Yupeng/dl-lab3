import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet
#class
class ResBlock(nn.Module):
    def __init__(self,inchannels, outchannels,strides,short = None):
        super(ResBlock,self).__init__()
        pre_res = [nn.Conv2d(inchannels, outchannels,3,stride=strides,padding=1),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(outchannels, outchannels, 3,stride=1,padding=1),
                        nn.BatchNorm2d(outchannels)]
        self.shortcut = short
        self.netres = nn.Sequential(*pre_res)
        self.relu = nn.ReLU()
    def forward(self, x):
        res = self.netres(x)
        short = x if self.shortcut == None else self.shortcut(x)
        return self.relu(res + short)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        layers = [
            nn.Conv2d(3,64,kernel_size=3,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1) #池化下采样，原图32*32，这一步完全可以不用就能获得比较好的感受野了
        ]
        group1 = self.layer_group(64,128,3,1)
        group2 = self.layer_group(128,256,3,2)
        group3 = self.layer_group(256,256,4,2)
        group4 = self.layer_group(256,256,3,2)
        layers.append(group1)
        layers.append(group2)
        layers.append(group3)
        layers.append(group4)
        self.layers = nn.Sequential(*layers)
        self.classfy = nn.Sequential(nn.Linear(256 * 2 * 2,128),nn.ReLU(inplace=True), nn.Linear(128,10))
    def layer_group(self, inchannels, outchannels, blocknum, downsample):
        shortcut = nn.Sequential(nn.Conv2d(inchannels, outchannels,kernel_size=1, stride=downsample),
                                 nn.BatchNorm2d(outchannels))
        layers = []
        downsample = ResBlock(inchannels, outchannels, downsample,shortcut)
        layers.append(downsample)
        for i in range(1,blocknum):
            layers.append(ResBlock(outchannels, outchannels,1))

        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        out = out.view([out.shape[0], -1])
        out = self.classfy(out)
        return out



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



#functions
def Criterion():
    return nn.CrossEntropyLoss()





def definenet(opt):
    class_str = opt.model
    CLASS = eval(class_str)()
    return CLASS
# a = definenet('ResNet34')
# print(a)