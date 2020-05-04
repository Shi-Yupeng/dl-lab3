from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.001, help='学习率，这里使用adam优化器')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='终端输出展示步长')
        self.isTrain = True
        return parser