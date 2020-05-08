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
        parser.add_argument('--optimizer', type=str, default='ADAM',
                            help='优化器 [SGD,SGD-Momentum,RMSProp,ADAM]')
        parser.add_argument('--fp16', action='store_true',
                            help='是否使用16-bit替代32-bit')
        parser.add_argument('--opt_level', type=str, default='O1',
                            help='半精度模式 [O0, O1, O2, 03]')
        self.isTrain = True
        return parser