import argparse
import torch
class BaseOptions():
    def __init__(self):
        pass
    def initialize(self, parser):
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. CPU使用-1')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存目录')
        parser.add_argument('--model', type=str, default='ResNet',
                            help='训练模型 vgg or resnet')
        parser.add_argument('--num_threads', default=4, type=int, help='数据读取线程')
        parser.add_argument('--batch_size', type=int, default=64, help=' batch size')
        parser.add_argument('--datashuffle', action='store_true', help=' 是否随机')
        parser.add_argument('--loadepoch', type=int, default=0,
                            help='需要加载的模型')

        parser.add_argument('--test_results_dir', type=str, default='./results/', help='可视化模型目录')
        parser.add_argument('--root', type=str, default='./dataset/cifar10/', help='数据集目录')
        return parser
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        opt = parser.parse_args()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt