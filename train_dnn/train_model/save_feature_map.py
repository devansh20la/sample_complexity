#
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

def main(args):
    hf = h5py.File(args.data_dir + "{}_{}_feat.hdf5".format(args.dtype, args.mtype), 'w')

    if 'imagenet' in args.dtype:
        train_size_list = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
    else:
        train_size_list = []

    for train_size in train_size_list:

        args.train_size = train_size
        args.cp_dir="{0}/checkpoints/{1}/{2}/{2}_{3}".format(args.dir, args.dtype, args.mtype, args.train_size)

        dset_loaders = tf.get_loader(args, training=True)
        model = tf.get_model(args)

        if args.use_cuda:
            model = model.cuda()

        state = torch.load(args.cp_dir)
        model.load_state_dict(state['model'])

        # classifier = list(model.classifier)
        # classifier.pop()
        # classifier.pop()
        # model.classifier = nn.Sequential(*classifier)

        model.eval()

        for phase in ["train", "val"]:
            hf_phase = hf.create_group("{}/{}".format(args.train_size, phase))
                        
            out_features = torch.zeros((len(dset_loaders[phase].dataset), args.feature_dim))
            out_targets = torch.zeros((len(dset_loaders[phase].dataset), 1))
            batch = 0

            for inp_data in tqdm(dset_loaders[phase]):

                inputs = inp_data['img']
                targets = inp_data['target']

                if args.use_cuda:
                    inputs = inputs.cuda()

                with torch.no_grad():
                    outputs = model(inputs)

                out_features[batch:batch + outputs.shape[0], :] = outputs.reshape(-1, args.feature_dim)
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
    else:
        quit()
        
    args.use_cuda = torch.cuda.is_available()

    main(args)
