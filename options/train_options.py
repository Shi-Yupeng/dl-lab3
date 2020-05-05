from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.001, help='学习率，这里使用adam优化器')
        parser.add_argument('--epoches', type=int, default=20,
                            help='训练次数')
        parser.add_argument('--print_freq', type=int, default=3,
                            help='终端输出展示步长')
        parser.add_argument('--save_freq', type=int, default=10,
                            help='存储结果步长')
        parser.add_argument('--lr_decay_iters', type=int, default=1, help='学习率更新时间')
        parser.add_argument('--lr_policy', type=str, default='none',
                            help='学习率更新准则')
        parser.add_argument('--continue_train', action='store_true', help=' 是否续训')
        parser.add_argument('--visual_module', type=str, default='./checkpoints/runs/',
                            help='可视化目录')
        self.isTrain = True
        return parser