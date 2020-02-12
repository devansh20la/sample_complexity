
# VERY IMPORTANT
#
# MAKE SURE TO FIRST CHANGE THE MODEL FILE SO THAT IT OUTPUTS FEATURE MAPS
# RATHER THAN THE CLASS LABELS
#
# VERY IMPORTANT
#
import torch
import torch.nn as nn
import os
import random
import utils.util_funcs as uf
import numpy as np
import train_funcs as tf
import copy
from tqdm import tqdm
import h5py
import argparse
from args import get_args
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

        self.bottleneck = bottleneck(args.feature_dim, args.dim, args.feature_dim, args.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.bottleneck(x)
        return x


def main(args):
    path = "data/cifar10/"
    hf = h5py.File(path + "new_bottleneck_{}_{}_feat.hdf5".format(args.dtype, args.mtype), 'w')
    args.cp_dir = "checkpoints/cifar10_bottleneck/resnet18/dim_2/"

    for dim in [2]:
        args.dim = dim 

        # if args.dtype == "cifar10":
        #     if args.mtype == "cifar_resnet18":
        #         args.cp_dir = os.path.join("checkpoints/cifar10_bottleneck/resnet18/dim_{}/run1/best_model.pth.tar".format(dim))

        #     elif args.mtype == "cifar_vgg16":
        #         args.cp_dir = os.path.join("checkpoints/cifar10_bottleneck/vgg16/dim_{}/run1/best_model.pth.tar".format(dim))

        # elif args.dtype == "cifar100":
        #     if args.mtype == "cifar_resnet18":
        #         args.cp_dir = os.path.join("checkpoints/cifar100_bottleneck/resnet18/dim_{}/run1/best_model.pth.tar".format(dim))

        #     elif args.mtype == "cifar_vgg16":
        #         args.cp_dir = os.path.join("checkpoints/cifar100_bottleneck/vgg16/dim_{}/run1/best_model.pth.tar".format(dim))

        # elif args.dtype == "mnist":
        #     if args.mtype == "lenet":
        #         pass
        #         # args.cp_dir = os.path.join("checkpoints/mnist_bottleneck/dim_{}/run0/best_model.pth.tar".format(dim))

        # elif args.dtype == "imagenet":
        #     if args.mtype == "imagenet_resnet50":
        #         args.cp_dir = os.path.join("checkpoints/imagenet_bottleneck/dim_{}/run0/best_model.pth.tar".format(dim))

        # else:
        #     quit()

        for args.train_size in [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]:

            dset_loaders = tf.get_loader(args, training=True)
            model = bottle_model(args)

            if args.use_cuda:
                # model = nn.DataParallel(model)
                model = model.cuda()

            state = torch.load("{}/{}/run0/best_model.pth.tar".format(args.cp_dir, args.train_size))
            model.load_state_dict(state['model'])

            classifier = list(model.bottleneck.fc)
            classifier.pop()
            classifier.pop()
            classifier.pop()
            model.bottleneck.fc = nn.Sequential(*classifier)

            model.eval()


            for phase in ["train","val"]:
                hf_phase = hf.create_group("{}/{}".format(args.train_size, phase))
                            
                out_features = torch.zeros((len(dset_loaders[phase].dataset), args.dim))
                out_targets = torch.zeros((len(dset_loaders[phase].dataset), 1))
                batch = 0

                for inp_data in tqdm(dset_loaders[phase]):

                    inputs = inp_data['img']
                    targets = inp_data['target']

                    if args.use_cuda:
                        inputs = inputs.cuda()

                    with torch.no_grad():
                        outputs = model(inputs)

                    out_features[batch:batch + outputs.shape[0], :] = outputs.reshape(-1, args.dim)
                    out_targets[batch:batch + outputs.shape[0], :] = targets.reshape(-1, 1)
                    batch += outputs.shape[0]

                hf_phase.create_dataset("features", data=out_features.numpy())
                hf_phase.create_dataset("targets", data=out_targets.numpy())
        hf.close()


if __name__ == '__main__':
    args = get_args()

    if "resnet18" in args.mtype:
        args.feature_dim = 512
    elif "vgg16" in args.mtype:
        args.feature_dim = 4096
    elif "lenet" in args.mtype:
        args.feature_dim = 84
    elif "imagenet_resnet50" in args.mtype:
        args.feature_dim = 2048
    else:
        quit()
        
    args.use_cuda = torch.cuda.is_available()

    main(args)
