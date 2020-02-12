import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as base_transforms
import torch
import sys
sys.path.insert(0, '../')
import utils.util_funcs as uf
from torchvision.transforms import functional as F


class ImageFolder(Dataset):

	def __init__ (self,root_dir,csv_file,training,transforms,use_bw_mask=True):
		self.data = pd.read_csv(csv_file, header=None)
		self.root_dir = root_dir
		self.training = training
		self.transforms = transforms
		self.use_bw_mask = use_bw_mask
		self.normalize = base_transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		self.to_tensor = base_transforms.ToTensor()


	def pil_loader(self, path):
	    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	    with open(path, 'rb') as f:
	        img = Image.open(f)
	        return img.convert('RGB')

	def pil_loader_ann(self, path):
	    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	    with open(path, 'rb') as f:
	        img = Image.open(f)
	        return img.convert('L')

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		imgname = os.path.join(self.root_dir,self.data.iloc[idx,0])
		image = self.pil_loader(imgname)
		label = torch.tensor(list(map(int,list(self.data.ix[idx,1:])))).type(torch.FloatTensor)

		if self.training == True:
			image_seg = self.pil_loader_ann(os.path.join(self.root_dir,self.data.iloc[idx,0].split('.')[0] + '_ann.jpg'))

			for t in self.transforms:
				image, image_seg = t(image, image_seg)

			image = self.normalize(self.to_tensor(image))	
			image_seg = self.to_tensor(image_seg)
		else:
			image_seg = None
			
			for t in self.transforms:
				if type(t) == base_transforms.TenCrop:
					mul_crops = True
				else:
					mul_crops = False

				image = t(image)
			
			if mul_crops:
				lam = base_transforms.Lambda(lambda crops: torch.stack([self.normalize(self.to_tensor(crop)) for crop in crops]))
				image = lam(image)
			else:
				image = self.normalize(self.to_tensor(image))			

		if self.training:
			if not self.use_bw_mask :
				image_seg = image_seg.expand(image.size())	

			sample = {'img': image, 'seg':image_seg, 'label': label, 'path': self.data.ix[idx,0]}
		else:
			sample = {'img': image, 'label': label, 'path': self.data.ix[idx,0]}
		return sample
