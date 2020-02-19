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
    hf = h5py.File(args.data_dir + "bottleneck_{}_{}_feat.hdf5".format(args.dtype, args.mtype), 'w')
    
    for dim in [1,2,3,4,5,6]:
        args.dim = dim
        args.cp_dir = "{0}/checkpoints/{1}/{2}/{2}_{3}_dim_{4}".format(args.dir, args.dtype, args.mtype, args.train_size, args.dim) 

        dset_loaders = tf.get_loader(args, training=True)
        model = bottle_model(args)

        if args.use_cuda:
            # model = nn.DataParallel(model)
            model = model.cuda()

        state = torch.load("{0}/run0/{1}".format(args.cp_dir, 'best_model.pth.tar'))
        model.load_state_dict(state['model'])

        classifier = list(model.bottleneck.fc)
        classifier.pop()
        classifier.pop()
        classifier.pop()
        model.bottleneck.fc = nn.Sequential(*classifier)

        model.eval()


        for phase in ["train","val"]:
            hf_phase = hf.create_group("{}/{}".format(args.dim, phase))
                        
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
        print("problem with input args")
        quit()
        
    args.use_cuda = torch.cuda.is_available()

    main(args)
