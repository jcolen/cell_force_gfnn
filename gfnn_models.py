import numpy as np
import pandas as pd
import torch
import os

import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from gfnn_data_processing import *
from torchvision.transforms import Compose

import warnings
warnings.filterwarnings('ignore')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
			padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sin(x)
        x = self.conv3(x)
        return x


mtwopii = -2.0j * np.pi		   
class ClebschGFNN(nn.Module):
	'''
	GFNN Model with Clebsch decomposition to predict forces
	'''
	def __init__(self, size=(256, 256), pad=0, **kwargs):
		super(ClebschGFNN, self).__init__()
		self.size = size
		self.pad = pad
		self.kernel = nn.Parameter(torch.empty((3, size[0]+pad, size[1]+pad), 
									           dtype=torch.cfloat), 
								   requires_grad=True)
		nn.init.xavier_uniform_(self.kernel)

		self.conv_in = nn.Sequential(
			ConvBlock(1, 64,  kernel_size=3),
			ConvBlock(64, 64, kernel_size=3),
			nn.Conv2d(64, 3, kernel_size=1),
		)
		
		q = fft.fftfreq(size[0]+self.pad)
		self.q = torch.nn.Parameter(
			torch.stack(torch.meshgrid(q, q, indexing='ij'), dim=0),
			requires_grad=False)

	def get_kernel(self):
		return self.kernel.exp()

	def forward(self, x):
		b, c, h, w = x.shape
		x = self.conv_in(x)

		pad_size = [h + self.pad, w + self.pad]

		xq = torch.fft.fft2(x, s=pad_size)
		xq = torch.einsum('jyx,bjyx->bjyx', self.get_kernel(), xq)
		
		grad_phi = fft.ifft2(mtwopii * self.q * xq[:, 0:1], s=pad_size).real
		psi = fft.ifft2(xq[:, 1:2], s=pad_size).real
		grad_chi = fft.ifft2(mtwopii * self.q * xq[:, 2:3], s=pad_size).real

		x = grad_phi + psi * grad_chi
		x = x[..., :h, :w] #Crop away spoiled terms

		return x
	
	def get_transform(self, cell_crop=800, downsample=2, nmax=50, **kwargs):
		transform = Compose([
			CellCrop(cell_crop, pad_type=0),
			Threshold('F_ml', threshold=0.5, rescale=1),
			Downsample(downsample),
			ToTensor(),
			FourierCutoff(nmax=nmax, key='zyxin'),
			GetXY(),
		])
		return transform

class RealGFNN(nn.Module):
	'''
	More efficient GFNN using rfft instead of fft
	'''
	def __init__(self, size=256, pad=0, **kwargs):
		super(RealGFNN, self).__init__()
		self.size = size
		self.pad = pad
		self.kernel = nn.Parameter(torch.empty((3, size+pad, (size+pad)//2+1), dtype=torch.cfloat),
								   requires_grad=True)
		nn.init.xavier_uniform_(self.kernel)

		self.conv_in = nn.Sequential(
			ConvBlock(1, 64,  kernel_size=3),
			ConvBlock(64, 64, kernel_size=3),
			nn.Conv2d(64, 3, kernel_size=1),
		)
		
		q = fft.fftfreq(size+self.pad)
		q = torch.stack(torch.meshgrid(q, q), dim=0)
		q = q[:, :self.kernel.shape[1], :self.kernel.shape[2]]
		self.q = torch.nn.Parameter(q, requires_grad=False)

	def get_kernel(self):
		return self.kernel.exp()

	def forward(self, x):
		b, c, h, w = x.shape
		x = self.conv_in(x)

		pad_size = [h + self.pad, w + self.pad]

		xq = torch.fft.rfft2(x, s=pad_size)
		xq = torch.einsum('jyx,bjyx->bjyx', self.get_kernel(), xq)
		
		grad_phi = fft.irfft2(mtwopii * self.q * xq[:, 0:1], s=pad_size)
		psi = fft.irfft2(xq[:, 1:2], s=pad_size)
		grad_chi = fft.irfft2(mtwopii * self.q * xq[:, 2:3], s=pad_size)

		x = grad_phi + psi * grad_chi
		x = x[..., :h, :w] #Crop away spoiled terms

		return x
	
	def get_transform(self, cell_crop=800, downsample=2, nmax=50, **kwargs):
		transform = Compose([
			CellCrop(cell_crop, pad_type=0),
			Threshold('F_ml', threshold=0.5, rescale=1),
			Downsample(downsample),
			ToTensor(),
			FourierCutoff(nmax=nmax, key='zyxin'),
			GetXY(),
		])
		return transform

class RealCoulombGFNN(RealGFNN):
	def forward(self, x):
		b, c, h, w = x.shape
		x = self.conv_in(x)

		pad_size = [h + self.pad, w + self.pad]

		xq = torch.fft.rfft2(x, s=pad_size)
		xq = torch.einsum('jyx,bjyx->bjyx', self.get_kernel(), xq)
		
		psi = fft.irfft2(xq[:, 1:2], s=pad_size)
		grad_chi = fft.irfft2(mtwopii * self.q * xq[:, 2:3], s=pad_size)

		x = psi * grad_chi
		x = x[..., :h, :w] #Crop away spoiled terms

		return x
