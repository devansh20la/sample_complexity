import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision import transforms as base_transforms
import torch
# from matplotlib import pyplot as plt
# from torchvision import utils as vutils 
# to_pil = transforms.ToPILImage()
import numpy as np 
# from syn_to_class import classes


class ImageFolder(data.Dataset):
	"""`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
	Args:
		root (string): Root directory where images are downloaded to.
		annFile (string): Path to json annotation file.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.ToTensor``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""

	def __init__(self, root, annFile, training, transforms=None, target_transform=None):
		from pycocotools.coco import COCO
		self.root = root
		self.coco = COCO(annFile)
		self.ids = list(self.coco.imgs.keys())
		self.transforms = transforms
		self.target_transform = target_transform
		self.normalize = base_transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		self.to_tensor = base_transforms.ToTensor()
		self.training = training
		self.count = 0
	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
		"""
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		image_seg = np.zeros((image.size[1],image.size[0]))

		cats = []
		for i in anns:
			image_seg += coco.annToMask(i)
			cats.append(i['category_id']-1)

		cats = np.unique(cats)
		label = np.zeros(91)

		for i in cats:
			label[i] = 1

		image_seg = Image.fromarray(np.uint8(image_seg*255)).convert('1')
		label = torch.tensor(label).type(torch.FloatTensor)

		if self.training == True:

			for t in self.transforms:
				image, image_seg = t(image, image_seg)

			image = self.normalize(self.to_tensor(image))	
			image_seg = self.to_tensor(image_seg)
			sample = {'img': image, 'seg':image_seg, 'label': label, 'path': path}

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

			sample = {'img': image, 'label': label, 'path': path}		
						
		return sample

	def __len__(self):
		return len(self.ids)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str

if __name__ == '__main__':

	def imshow(tensor):
		tensor = vutils.make_grid(tensor, normalize=True)
		tensor = to_pil(tensor)
		plt.imshow(tensor)

	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225])

	trans = transforms.Compose([transforms.Resize(256),
						transforms.RandomCrop(224),
						transforms.ToTensor(),
						normalize])
					   
	dsets = CocoDetection('val2017/', 'instances_val2017.json', transform=trans)
	d_loader = torch.utils.data.DataLoader(dsets,batch_size=1, num_workers=0,shuffle=True)
	for data in d_loader:
		inputs = data[0]
		target = data[1]
		break

