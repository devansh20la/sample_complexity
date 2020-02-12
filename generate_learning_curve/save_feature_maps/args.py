import argparse
import os
import torch


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--ms', type=int, default=5)
    parser.add_argument('--mtype', type=str, required=True)

    parser.add_argument('--dtype', type=str, required=True, help='Data type')
    parser.add_argument('--ep', type=int, default=250, help='Epochs')

    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default = 128, help='batch size')
    parser.add_argument('--m', type=float, default=0.9, help='momentun')

    parser.add_argument('--img_per_class', type=int, default=None)
    parser.add_argument('--train_size', type=float, default=1.0)

    parser.add_argument('--print_freq', type=int, default=50)

    
    args = parser.parse_args(*args)

    args.dir = "."
    args.cp_dir = "{0}/checkpoints/{1}/{2}/{2}_{3}".format(args.dir, args.dtype, args.mtype, args.train_size)
    
    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [150, 250]
        args.data_dir = "{}/data/cifar/".format(args.dir)

    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [150, 250]
        args.data_dir = "{}/data/cifar/".format(args.dir)

    elif args.dtype == 'imagenet':
        args.num_classes = 1000
        args.milestones = [30, 60, 90]
        args.data_dir = "{}/data/imagenet/".format(args.dir)

    elif args.dtype == 'mnist':
        args.num_classes = 10
        args.data_dir = "{}/data/mnist/".format(args.dir)

    elif args.dtype == 'udacity':
        args.num_classes = None
        args.milestones = [70, 120, 150]
        args.data_dir = "{}/data/udacity/".format(args.dir)

    args.use_cuda = torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()
