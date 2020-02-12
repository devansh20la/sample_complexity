import numpy as np
from collections import defaultdict
import os
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch
import csv
import random
import scipy.io
from sklearn.model_selection import train_test_split


class SyntheticDataset():
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return self.samples[0].shape[0]

    def __getitem__(self, idx):

        img = self.samples[0][idx, :, :, :]
        target = self.samples[1][idx, :]

        img = torch.from_numpy(img).type(torch.FloatTensor)

        if self.transform is not None:
            img = self.transform(img)

        target = torch.from_numpy(target)
        sample = {'img': img, 'target': target.item()}

        return sample
