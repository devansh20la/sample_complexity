import numpy as np
from collections import defaultdict
import os
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch
import csv


def make_dataset(root):

	samples = []
	class_vector = []
	count = 0

	for folder_name in os.listdir(root):
		if not os.path.isdir(os.path.join(root, folder_name)): continue
		fopen = open(os.path.join(root, folder_name,'video_labels.csv'),'r')
		fopen = csv.reader(fopen, delimiter=',')
		next(fopen)
		
		for i,line in enumerate(fopen,1):
			item = (os.path.join(root, folder_name, 'frames/{:06d}.png'.format(i)), float(line[3])*40)

			if np.abs(item[1]) < 0.034:
				count += 1
				if count < 4000:
					samples.append(item)
					class_vector.append(item[1])
			else:
				samples.append(item)
				class_vector.append(item[1])
			
	class_vector = np.array(class_vector)
	bins = np.linspace(-0.6, 0.8, 10)
	class_vector = np.digitize(class_vector, bins)

	samples = [(i[0],j) for i,j in zip(samples, class_vector)]
	
	return samples, class_vector


class SyntheticDataset():
	"""NVIDIA Synthetic dataset."""
	def __init__(self, samples, transform=None):

		self.transform = transform
		self.samples = samples
			
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img_name = self.samples[idx][0]
		target = torch.Tensor([self.samples[idx][1]])

		img = Image.open(img_name)

		if self.transform is not None:
			img = self.transform(img)

		sample = {'img': img, 'taget': target}
		
		return sample