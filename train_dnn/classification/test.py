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


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def main(args):
    print("....Initializing data sampler.....")

    dset_loaders = tf.get_loader(args, training=False)
    model = tf.get_model(args)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    raw_state = copy.deepcopy(model.state_dict())
    all_errors = np.zeros((args.rep_times, len(args.model_names)), dtype='float32')

    for j, model_name in enumerate(tqdm(args.model_names)):
        error_meter = uf.AverageMeter()

        if model_name is not None:
            run_files = sorted(glob.glob1(os.path.join(args.cp_dir, model_name), 'run*'))
        else:
            run_files = [None] * args.rep_times

        for i, run in enumerate(run_files):
            if model_name is not None:
                state = torch.load(os.path.join(args.cp_dir, model_name, run, 'best_model.pth.tar'))
                model.load_state_dict(state['model'])
            else:
                model.load_state_dict(raw_state)

            for batch_idx, inp_data in enumerate(dset_loaders['test'], 1):

                inputs = inp_data['img']
                targets = inp_data['target']

                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                with torch.no_grad():       # compute output
                    outputs = model(inputs)

                batch_err = 1.0 - uf.accuracy(outputs, targets, topk=(1,))[0]/100
                
                if 'udacity' in args.dtype:
                    batch_err = torch.abs(outputs - targets)
                    batch_err = torch.sum(batch_err > 0.1).type(torch.FloatTensor)/inputs.size(0)
                
                error_meter.update(batch_err, inputs.size(0))

            all_errors[i, j] = error_meter.avg
    scipy.io.savemat(args.save_path, dict(y=all_errors))


if __name__ == '__main__':

    args = get_args()
    args.use_cuda = torch.cuda.is_available()

    if args.dtype == 'cifar100':
        args.save_path = 'results/cifar100/{}_cifar100_errors.mat'.format(args.mtype)
        args.model_names = sorted(glob.glob1(args.cp_dir, '*cifar*'))
        args.model_names = [None, *args.model_names]
        args.rep_times = 5
        print(args.model_names)

    elif args.dtype == 'cifar10':
        args.save_path = 'results/cifar10/{}_cifar10_errors.mat'.format(args.mtype)
        args.model_names = sorted(glob.glob1(args.cp_dir, '*cifar*'))
        args.model_names = [None, *args.model_names]
        args.rep_times = 5
        print(args.model_names)

    elif args.dtype == 'imagenet':
        args.mtype = 'imagenet_resnet50'
        args.save_path = 'results/imagenet/{}_imagenet_errors.mat'.format(args.mtype)
        args.model_names = sorted(glob.glob1(args.cp_dir, '*imagenet*'))
        args.model_names = [None, *args.model_names]
        args.rep_times = 1
        print(args.model_names)

    elif args.dtype == 'mnist':
        args.model_names = sorted(glob.glob1(args.cp_dir, '*mnist*'))
        args.model_names = [None, *args.model_names]
        print(args.model_names)
        args.save_path = "results/mnist/{}_errors.mat".format(args.mtype)
        args.rep_times = 1
        args.num_classes = 10
    
    elif args.dtype == 'udacity':
        args.model_names = sorted(glob.glob1(args.cp_dir, '*udacity*'))
        args.model_names = [None, *args.model_names]
        print(args.model_names)
        args.save_path = "results/udacity/{}_errors.mat".format(args.mtype)
        args.rep_times = 5

    main(args)
