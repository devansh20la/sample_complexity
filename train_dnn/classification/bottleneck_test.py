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

        # classifier = list(self.model.classifier)
        # classifier.pop()
        # classifier.pop()
        # self.model.classifier = nn.Sequential(*classifier)

        self.bottleneck = bottleneck(84, args.dim, 84, args.num_classes)
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

        state = torch.load(os.path.join(args.cp_dir, model_name, 'run2/best_model.pth.tar'))
        model.load_state_dict(state['model'])

        for batch_idx, inp_data in enumerate(tqdm(dset_loaders['test']), 1):

            inputs = inp_data['img']
            targets = inp_data['target']

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():       # compute output
                outputs = model(inputs)

            batch_loss = criterion(outputs, targets)

            batch_err = 100.0 - uf.accuracy(outputs, targets, topk=(1,))[0]
            error_meter.update(batch_err, inputs.size(0))

        all_errors[0, dim-1] = error_meter.avg
    scipy.io.savemat(args.save_path, dict(y=all_errors))


if __name__ == '__main__':

    args = get_args()
    args.use_cuda = torch.cuda.is_available()

    if args.dtype == 'cifar100':
        args.save_path = 'results/cifar100/bottleneck/vgg16_errors.mat'
        args.num_classes = 100
        create_path(args.save_path.split('vgg16_errors.mat')[0])

    elif args.dtype == 'cifar10':
        args.save_path = 'results/cifar10/bottleneck/vgg16_errors.mat'
        args.num_classes = 10
        create_path(args.save_path.split('vgg16_errors.mat')[0])


    elif args.dtype == 'mnist':
        args.save_path = "results/mnist/bottleneck/errors.mat"
        create_path(args.save_path.split('errors.mat')[0])
        args.num_classes = 10

    elif args.dtype == 'imagenet':
        args.save_path = "results/imagenet/bottleneck/errors.mat"
        create_path(args.save_path.split('errors.mat')[0])
        args.num_classes = 1000

    main(args)
