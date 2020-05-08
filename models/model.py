import torch
import os
from models import networks
from tensorboardX import SummaryWriter
import numpy as np
from torch.optim import lr_scheduler
class Model():
    def __init__(self, opt):
        self.writer = SummaryWriter(opt.visual_module + opt.model + '/')
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = opt.checkpoints_dir
        self.model_name = opt.model
        self.net = networks.definenet(opt).to(self.device)
        if self.isTrain:
            self.criterion = networks.Criterion().to(self.device)
            optimizers = {"SGD": torch.optim.SGD(self.net.parameters(),
                                             lr=opt.lr),
                          "SGD-Momentum": torch.optim.SGD(self.net.parameters(),
                                                      momentum=0.99,
                                                      lr=opt.lr),
                          "RMSProp": torch.optim.RMSprop(self.net.parameters(),
                                                     alpha=0.9),
                          'ADAM': torch.optim.Adam(self.net.parameters(),
                                               lr=opt.lr, betas=(0.9, 0.99))}
            self.optimizer = optimizers[opt.optimizer]

    def initialize(self, opt):

        if self.isTrain:
            if opt.lr_policy == 'step':
                scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
                self.scheduler = scheduler
            elif opt.lr_policy == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, threshold=0.01,
                                                           patience=3)
                self.scheduler = scheduler
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.loadepoch)
        self.print_networks()

    def update_learning_rate(self, opt):
        if opt.lr_policy == 'step':
            self.scheduler.step()
        elif opt.lr_policy == 'plateau':
            self.scheduler.step(self.loss)
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_input(self, imgs, labels):
        self.imgs = imgs.to(self.device)
        self.labels = labels.to(self.device)

    def forward(self):
        self.predict = self.net(self.imgs)

    def backward(self):
        self.loss = self.criterion(self.predict,self.labels)
        self.loss.backward()
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def eval(self):
        self.net.eval()

    def test(self, testset):
        self.eval()
        totals = 0
        correct = 0
        for data in testset:
            imgs, labels = data
            totals += labels.size(0)
            imgs = imgs.to(self.device)
            with torch.no_grad():
                predict = self.net(imgs)
                if torch.cuda.is_available():
                    _, predict = torch.max(predict.cpu().data, 1)
                else:
                    _, predict = torch.max(predict.data, 1)
            correct += (predict == labels).numpy().astype(np.int).sum()
        return correct / totals



    def compute_visuals(self,epoch, trainset,testset):
        self.train_acc = self.test(trainset)
        self.test_acc = self.test(testset)
        if torch.cuda.is_available():
            loss = self.loss.cpu().data.numpy()
        else:
            loss = self.loss.data.numpy()
        self.writer.add_scalars('scalar',{'train_loss': loss ,
                                          'train_acc': self.train_acc,
                                          'test_acc': self.test_acc},global_step=epoch)
        print('----------epoch{}----------'.format(str(epoch)))
        print('train_acc',self.train_acc)
        print('test_acc',self.test_acc)
        print('loss',loss)
        print('----------------------')

    def load_networks(self, epoch):
        load_filename = '%s_net_%s.pth' % (epoch, self.opt.model)
        load_path = os.path.join(self.save_dir, load_filename)
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.net.load_state_dict(state_dict)


    def save_networks(self, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, self.opt.model)
        save_path = os.path.join(self.save_dir, save_filename)
        if torch.cuda.is_available():
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.to(self.device)
        else:
            torch.save(self.net.cpu().state_dict(), save_path)


    def print_networks(self):
        print('---------- Networks initialized -------------')
        print(self.net)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.net.parameters())))
        print('-----------------------------------------------')