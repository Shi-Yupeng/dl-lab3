from options.train_options import TrainOptions

opt = TrainOptions().get_parser()
print(opt.batch_size)