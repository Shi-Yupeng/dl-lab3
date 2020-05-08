from models import model
import data
from options.train_options import TrainOptions


if __name__ == "__main__":
    opt = TrainOptions().get_parser()
    traindata = data.create_dataset(opt)
    testdata = data.Testdata(opt)
    model = model.Model(opt)
    model.initialize(opt)
    start = 1
    if opt.continue_train:
        start = opt.loadepoch
    for epoch in range(start, opt.epoches):
        for data in traindata:
            imgs, labels = data
            model.set_input(imgs, labels)
            model.optimize_parameters()
        if epoch % opt.print_freq == 0:
            model.compute_visuals(epoch,traindata,testdata)
        if epoch % opt.save_freq == 0:
            model.save_networks(epoch)
        if opt.lr_policy != 'none' and epoch % opt.lr_decay_iters == 0:
            model.update_learning_rate(opt)