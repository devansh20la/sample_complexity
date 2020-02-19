import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import random
import utils.util_funcs as uf
import time
import logging
import numpy as np
from args import get_args
import train_funcs as tf
import copy
from tqdm import tqdm
import scipy
import glob
from models import bottleneck


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

class bottle_model(nn.Module):

    def __init__(self, args):
        super(bottle_model, self).__init__()
        self.model = tf.get_model(args)

        if 'vgg' in args.mtype:
            classifier = list(self.model.classifier)
            classifier.pop()
            classifier.pop()
            self.model.classifier = nn.Sequential(*classifier)

        elif 'udacity' in args.mtype:
            regressor = list(self.model.regressor)
            regressor.pop()
            self.model.regressor = nn.Sequential(*regressor)

        self.bottleneck = bottleneck(args.feature_dim, args.dim, args.feature_dim, args.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.bottleneck(x)

        return x


def main(args):
    print("....Initializing data sampler.....")

    dset_loaders = tf.get_loader(args, training=False)
    all_errors = np.zeros((1, 6), dtype='float32')

    for dim in range(1,7):
        args.dim = dim
        model_name = "dim_{}".format(args.dim)

        model = bottle_model(args)
        # model = nn.DataParallel(model)
        model.eval()
        
        if args.use_cuda:
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        error_meter = uf.AverageMeter()

        state = torch.load("{0}_dim_{1}/run0/{2}".format(args.load_cp, int(dim), 'best_model.pth.tar'))
        model.load_state_dict(state['model'])

        for batch_idx, inp_data in enumerate(tqdm(dset_loaders['test']), 1):

            inputs = inp_data['img']
            targets = inp_data['target']

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():       # compute output
                outputs = model(inputs)

            batch_loss = criterion(outputs, targets)

            batch_err = 1.0 - uf.accuracy(outputs, targets, topk=(1,))[0]/100
            error_meter.update(batch_err, inputs.size(0))

        all_errors[0, dim-1] = error_meter.avg
    scipy.io.savemat(args.save_path, dict(y=all_errors))


if __name__ == '__main__':

    args = get_args()
    args.use_cuda = torch.cuda.is_available()
    args.load_cp = "{0}/checkpoints/{1}/{2}/{2}_{3}".format(args.dir, args.dtype, args.mtype, args.train_size)

    if args.dtype == 'cifar100':
        args.save_path = 'results/cifar100/bottleneck_{}_cifar100_errors.mat'.format(args.mtype)

    elif args.dtype == 'cifar10':
        args.save_path = 'results/cifar10/bottleneck_{}_cifar10_errors.mat'.format(args.mtype)

    elif args.dtype == 'imagenet':
        args.save_path = 'results/imagenet/bottleneck_{}_imagenet_errors.mat'.format(args.mtype)

    elif args.dtype == 'mnist':
        args.save_path = "results/mnist/bottleneck_{}_errors.mat".format(args.mtype)
    
    elif args.dtype == 'udacity':
        args.save_path = "results/udacity/bottleneck_{}_errors.mat".format(args.mtype)

    if "resnet18" in args.mtype:
        args.feature_dim = 512

    elif "vgg16" in args.mtype:
        args.feature_dim = 4096

    elif "lenet" in args.mtype:
        args.feature_dim = 84
        
    elif "resnet50" in args.mtype:
        args.feature_dim = 2048
    
    elif "udacity_cnn" in args.mtype:
        args.feature_dim = 1024
    else:
        quit()

    f = args.save_path.split('/')
    create_path("/".join(f[:-1]))
    main(args)
