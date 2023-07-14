import numpy as np
import pandas as pd
import torch
import os

import warnings
warnings.filterwarnings('ignore')

'''
Image processing functions
'''
class CellCrop(object):
	"""
	min_factor is the minimum factor that the image dimensions should be. I.e. if 16, then each dimension must be a multiple of 16
	"""
	def __init__(self, end_size, pad_type='zeros'):
		self.end_size = end_size
		self.pad_type = pad_type

	def __call__(self, sample):
		mask = sample['mask']
		t = np.max(np.nonzero(mask[0]), axis=1) # top right
		b = np.min(np.nonzero(mask[0]), axis=1) #bottom left

		cell_size = t-b
		if np.any(cell_size>=self.end_size):
			cell_size=np.ones_like(cell_size)*self.end_size # 960 max im size
			noname=1
		if np.any(cell_size == self.end_size):
			noname2=1
			cent = np.round(((t+b)/2)).astype(int)
			flexible_range = np.floor(((t-b)-self.end_size)/2)
			flexible_range = np.maximum(flexible_range, 0)
			pad_amt = self.end_size // 2
			cent += self.end_size // 2
			cent += [np.random.randint(-flexible_range[0], flexible_range[0]+1), np.random.randint(-flexible_range[1], flexible_range[1]+1)]

			pad = ((0,0), (pad_amt, pad_amt), (pad_amt,pad_amt))
			for key in sample:
				sample[key] = np.pad(sample[key], pad, mode='constant', constant_values=0) #pad first for boundary
				sample[key] = sample[key][:, cent[0]-pad_amt:cent[0]+pad_amt,cent[1]-pad_amt:cent[1]+pad_amt]
		else:
			for key in sample:
				sample[key] = sample[key][:, b[0]:t[0], b[1]:t[1]]

		pad = self.end_size - cell_size
		if np.any(pad<0): print(pad, cell_size)
		padL = np.floor(pad/2).astype(int)
		padR = np.ceil(pad/2).astype(int)

		pad = ((0,0),) + ((padL[0], padR[0]), (padL[1], padR[1]))

		if self.pad_type == 'zeros': kwargs={'mode': 'constant', 'constant_values': 0.}
		elif isinstance(self.pad_type, int):	kwargs = {'mode': 'constant', 'constant_values': self.pad_type}
		else:						 kwargs={'mode': 'reflect'}
		for key in sample:
			if key == 'mask': sample[key] = np.pad(sample[key], pad, mode=kwargs['mode'], constant_values=0)
			else: sample[key] = np.pad(sample[key], pad, **kwargs)

		try:
			assert(sample['mask'].shape[-2:]==(self.end_size, self.end_size))
		except:
			print(sample['mask'].shape)
			print(cell_size, t, b)
			print(np.unique(sample['mask']))
			print(noname, noname2)
			print(cent, pad_amt, pad)
		return sample

class Threshold(object):
	def __init__(self, key, threshold=0.5, rescale=1000):
		self.threshold = threshold
		self.rescale = rescale
		self.key = key

	def __call__(self, sample):
		sample[self.key] /= self.rescale
		thresh = np.linalg.norm(sample[self.key], axis=0) < self.threshold
		sample[self.key][:, thresh] = 0
		return sample

from skimage.measure import block_reduce
class Downsample(object):
	def __init__(self, factor):
		self.factor = factor

	def __call__(self, sample):
		for key in sample:
			sample[key] = block_reduce(sample[key], (1, self.factor, self.factor), np.mean)
		return sample

class ToTensor(object):
	def __call__(self, sample):
		for key in sample.keys():
			dtype = torch.bool if sample[key].dtype == bool else torch.float32
			sample[key] = torch.tensor(sample[key].copy(), dtype=dtype)
		return sample

'''
Further processing for GFNN
'''
class FourierCutoff(object):
	def __init__(self, nmax=60, key='zyxin'):
		self.nmax = nmax
		self.key = key
	def __call__(self, sample):
		fq = torch.fft.fft2(sample[self.key], dim=(-2, -1))
		q = torch.stack(torch.meshgrid(
				torch.fft.fftfreq(fq.shape[-2]),
				torch.fft.fftfreq(fq.shape[-1]), indexing='xy'), dim=-1)
		nmag = torch.linalg.norm(q, dim=-1) * q.shape[0]
		fq[..., nmag > self.nmax] = 0
		sample[self.key] = torch.fft.ifft2(fq, dim=(-2, -1)).real

		return sample

class GetXY(object):
	def __init__(self, x_key='zyxin', y_key='F_ml'):
		self.x_key = x_key
		self.y_key = y_key
	
	def __call__(self, sample):
		return sample[self.x_key], sample[self.y_key]


mtwopii = -2.0j * np.pi		   
class GetScalarTerms(object):
	def __init__(self):

		self.scalar_names = [
			r'\zeta',
			r'\left( \nabla \zeta \right)^2',
			r'\nabla^2 \zeta',
			r'\zeta^2',
			r'\zeta \left( \nabla \zeta \right)^2',
			r'\zeta \nabla^2 \zeta',
		]
				
	def __call__(self, sample):
		with torch.no_grad():
			x = sample['zyxin'][0]
			xq = torch.fft.fft2(x, dim=(-2, -1))
			L = x.shape[-1]
			r = torch.arange(L, dtype=torch.float) - L // 2
			r = torch.fft.ifftshift(r)
			r = torch.linalg.norm(torch.stack(torch.meshgrid(r, r, indexing='ij')), dim=0)

			q = torch.fft.fftfreq(L)
			q = torch.stack(torch.meshgrid(q, q, indexing='xy'), dim=0)

			gradx = torch.fft.ifft2(mtwopii * q * xq[None, :, :], dim=(-2, -1)).real
			dx2x = torch.fft.ifft2(mtwopii**2 * q[0]**2 * xq, dim=(-2, -1)).real
			dy2x = torch.fft.ifft2(mtwopii**2 * q[1]**2 * xq, dim=(-2, -1)).real

			base_mags = [
				x,
				gradx.pow(2).sum(dim=-3),
				dx2x + dy2x
			]

			charges = torch.stack([
				base_mags[0],
				base_mags[1],
				base_mags[2],
				base_mags[0]**2,
				base_mags[0] * base_mags[1],
				base_mags[0] * base_mags[2],
			])
			sample['charges'] = charges

			return sample


'''
Cell Dataset
'''
import h5py
class Dataset(torch.utils.data.Dataset):
	def __init__(self,
				 transform=None):
		super(Dataset, self).__init__()
		self.transform = transform
		self.dataframe = pd.read_csv('/home/jcolen/CellProject/data/cell_dataset.csv')
		self.data_path = '/home/jcolen/CellProject/data/cell_dataset.h5'
		self.dataset = None

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		if self.dataset is None:
			self.dataset = h5py.File(self.data_path, 'r')

		key = '%s/%03d' % (self.dataframe.cell[idx], self.dataframe.cell_idx[idx])
		sample = {k: self.dataset[key][k][()] for k in self.dataset[key]}

		if self.transform:
			sample = self.transform(sample)
		return sample
