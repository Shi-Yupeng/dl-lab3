from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--test_dir', type=str, default='test', help='准确率测试（训练集or测试集）')
        self.isTrain = False