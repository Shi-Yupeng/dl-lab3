import torch
import os
from models import model
import data
from options.test_options import TestOptions

if __name__ == "__main__":
    opt = TestOptions().get_parser()
    testdata = data.create_dataset(opt)
    model = model.Model(opt)
    model.initialize(opt)
    print('测试集准确率',model.test(testdata))

