import numpy as np
import pandas as pd
import torch
import os
import re
import glob
import gc
import warnings

from skimage.measure import block_reduce
from torchvision.transforms import Compose

from tqdm.auto import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import ElasticNet
import pysindy as ps

warnings.filterwarnings('ignore')

mtwopii = -2.0j * np.pi		   

class GetScalarTerms(object):
	scalar_names = [
		r'\zeta',
		r'\left( \nabla \zeta \right)^2',
		r'\nabla^2 \zeta',
		r'\zeta^2',
		r'\zeta \left( \nabla \zeta \right)^2',
		r'\zeta \nabla^2 \zeta',
	]
	
	def __init__(self, cutoff=60):
		self.cutoff = cutoff

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

			nmag = torch.linalg.norm(q * L, dim=-3)
			xq[..., nmag > self.cutoff] = 0.

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

		return {
			'zyxin': sample['zyxin'],
			'charges': charges,
			'Fml': sample['Fml'],
			'Fexp': sample['Fexp'],
			'mask': sample['mask'],
		}

log10 = np.log(10.)
def exp_decay(r, l=10):
	return np.exp(-r / l * log10)

class GetClebschTerms(object):
	dir_funcs = [
		lambda r: 1. / (r + 0.1j),
		lambda r: r,
		lambda r: torch.log(r + 0.1j),
		lambda r: exp_decay(r, 20),
		lambda r: exp_decay(r, 60),
	]
	dir_names=[
		'|x-r|^{-1}',
		'|x-r|',
		'\log|x-r|',
		'e^{-r / \ell_1}',
		'e^{-r / \ell_2}',
	 ]
	
	@staticmethod
	def get_term_names(scalar_names):
		potential_names = []
		for i in range(len(scalar_names)):
			for j in range(len(GetClebschTerms.dir_names)):
				potential_names.append('\\big( %s \\star %s \\big)' % \
									  (scalar_names[i], GetClebschTerms.dir_names[j]))
		
		phi_names = ['\\nabla %s' % pn for pn in potential_names]
		xichi_names = []
		for i in range(len(potential_names)):
			for j in range(len(potential_names)):
				xichi_names.append('%s \\nabla %s' % (potential_names[i], potential_names[j]))
		
		return xichi_names + phi_names

	def __call__(self, sample):
		with torch.no_grad():
			charges = sample['charges']
			L = charges.shape[-1]
			r = torch.arange(L, dtype=torch.float) - L // 2
			r = torch.fft.ifftshift(r)
			r = torch.linalg.norm(torch.stack(torch.meshgrid(r, r, indexing='ij')), dim=0)

			q = torch.fft.fftfreq(L)
			q = torch.stack(torch.meshgrid(q, q, indexing='xy'), dim=0)

			gr = torch.stack([df(r) for df in self.dir_funcs])
			cq = torch.fft.fft2(charges, dim=(-2, -1))
			gq = torch.fft.fft2(gr, dim=(-2, -1))

			terms = []
			#Generate terms which could appear in Clebsch potentials
			potentials_q = torch.einsum('iyx,jyx->ijyx', cq, gq)
			potentials_q = potentials_q.reshape([-1, *potentials_q.shape[-2:]])
						
			#Generate vectorial_terms from Clebsch potentials
			potentials = torch.fft.ifft2(potentials_q, dim=(-2, -1)).real
			grad_potentials = torch.fft.ifft2(mtwopii * q * potentials_q[:, None], dim=(-2, -1)).real
						
			nl_terms = torch.einsum('ayx,biyx->abiyx', potentials, grad_potentials)
			nl_terms = nl_terms.reshape([-1, 2, *potentials.shape[-2:]])
			terms = torch.cat([nl_terms, grad_potentials], dim=0)
 
		return {
			'zyxin': sample['zyxin'],
			'terms': terms,
			'Fml': sample['Fml'],
			'Fexp': sample['Fexp'],
			'mask': sample['mask'],
		}	 

'''
***************************************************

Extra Image Processing Functions

***************************************************
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
	def __init__(self, key, threshold=0.3, rescale=1000):
		self.threshold = threshold
		self.rescale = rescale
		self.key = key

	def __call__(self, sample):
		sample[self.key] /= self.rescale
		thresh = np.linalg.norm(sample[self.key], axis=0) < self.threshold

		sample[self.key][:, thresh] = 0
		return sample
	
class ToTensor(object):
	def __call__(self, sample):
		for key in sample.keys():
			sample[key] = torch.tensor(sample[key].copy(), 
				dtype=torch.bool if key == 'mask' else torch.float32)
		return sample
	
class Downsample(object):
	def __init__(self, factor):
		self.factor = factor

	def __call__(self, sample):
		for key in sample:
			sample[key] = block_reduce(sample[key], (1, self.factor, self.factor), np.mean)
		return sample

'''
Default Image transform
'''

transform = Compose([
	CellCrop(800, pad_type=0),
	Threshold('Fexp', threshold=0.5, rescale=1),
	Threshold('Fml', threshold=0.5, rescale=1),
	Downsample(2),
	ToTensor(),
	GetScalarTerms(cutoff=50),
	GetClebschTerms(),
])

'''
Dataset class
'''

class CellPredictionsDataset(torch.utils.data.Dataset):
	def __init__(self,
				 mask_dir='/project/vitelli/cell_stress/TractionData_All_16kpa_new',
				 pred_dir='/project/vitelli/cell_stress/ForcePrediction_All_16kpa_new/AM_all_fixed_NormedF_dataset2/testsplit_0',
				 cell='11_cell_1',
				 transform=transform):
		super(CellPredictionsDataset, self).__init__()
		self.transform = transform

		self.dataframe = pd.DataFrame(columns=['cell', 'idx', 'filename'])
		cells, idxs = [], np.zeros(0, dtype=int)
		subdir = os.path.join(mask_dir, cell)
		
		inputfiles = self.collect_folder(os.path.join(pred_dir, cell, 'inputs_*'))
		targetfiles = self.collect_folder(os.path.join(pred_dir, cell, 'outputs_*'))
		predfiles = self.collect_folder(os.path.join(pred_dir, cell, 'predictions_*'))
		
		self.dataframe = pd.DataFrame({'cell': [cell,] * len(inputfiles), 
									   'zyxin': inputfiles,
									   'Fexp': targetfiles,
									   'Fml': predfiles})
		self.movies = np.concatenate(
			[self.get_file_index(i) for i in range(len(self.dataframe))])
		self.movies[:, 1:3] = np.stack([
			self.movies[:, 1] * np.cos(self.movies[:, 2]),
			self.movies[:, 1] * np.sin(self.movies[:, 2])], axis=1)
		gc.collect()

	def get_file_index(self, i):
		zyxin = np.load(self.dataframe.zyxin[i], mmap_mode='r')
		Fml = np.load(self.dataframe.Fml[i], mmap_mode='r')
		Fexp = np.load(self.dataframe.Fexp[i], mmap_mode='r')
		return np.concatenate((zyxin, Fml, Fexp), axis=1)
		
	def collect_folder(self, dirpath):
		filenames = [f for f in glob.glob(os.path.join(dirpath, '*.npy'))]
		inds = [list(map(int, re.findall(r'\d+', f)))[-1] for f in filenames]
		return [f for _, f in sorted(zip(inds, filenames))]
	
	def __getitem__(self, idx):
		sample = {
			'zyxin': self.movies[idx, 0, None],
			'mask': self.movies[idx, 0, None] != 0,
			'Fml': self.movies[idx, 1:3],
			'Fexp': self.movies[idx, 3:]
		}
		
		sample = self.transform(sample)
		return sample

	def __len__(self):
		return len(self.dataframe)
	
	def collect_batch(self, indices):
		batch = [self.__getitem__(i) for i in indices]
		sample = {}
		for key in batch[0]:
			sample[key] = np.stack([b[key] for b in batch])
		return sample


# # Fitting on neural network outputs



def revert_shape(v, mask):
	vr = np.zeros([2, *mask.shape[-2:]])
	vr[mask.repeat(2, axis=-3)] = v[:, 0]
	return vr

def predict_frame(models, dataset, X_scaler, y0_scaler, index):
	batch = dataset[index]
	X, mask = batch['terms'], batch['mask'].bool().numpy()
	mout = mask.repeat(2, axis=-3)
	mse_ml_exp = np.mean(np.power(batch['Fml'] - batch['Fexp'], 2).numpy()[mout])
	X = X[:, mask.repeat(X.shape[-3], axis=-3)].T
	X = X_scaler.transform(X)
	mse_mo_exp = []
	num_terms = []
	for model in models:
		complexity = np.sum(model.coefficients() != 0)
		num_terms.append(complexity)
		y = model.predict(X)
		y = revert_shape(y0_scaler.inverse_transform(y), mask)
		mse_mo_exp.append(np.mean(np.power(y - batch['Fexp'].numpy(), 2)[mout]))
	out = pd.DataFrame({'num_terms': num_terms, 'mse_mo': mse_mo_exp})
	out['mse_ML'] = mse_ml_exp
	out['frame'] = index
	return out

def evaluate_cell(cell):
	X_scaler = StandardScaler()
	y0_scaler = StandardScaler()
	train_points = 50000
	alphas = [1e1, 5e0, 1e0, 
			  8e-1, 7e-1, 6e-1, 5e-1, 1e-1, #Empirically see a drop off around here
			  8e-2, 7e-2, 6e-2, 5e-2, 1e-2, #Empirically see a drop off around here
			  5e-3, 1e-3, 
			  5e-4, 1e-4]

	dataset = CellPredictionsDataset(cell)
	print(cell, len(dataset), flush=True)

	'''
	Because otherwise the memory loads get crazy, choose 1 split and 
	run the program 5 times
	'''
	outname = f'data/adding_terms/{cell}.csv'
	if os.path.exists(outname):
		test_results = pd.read_csv(outname, index_col=0)
	else:
		test_results = pd.DataFrame()
	#kf = KFold(n_splits=5, shuffle=True)
	#Note that we reverse the order - we train on 1/5 of each cell and test on 4/5 of each cell

	#N = 1 case, cannot use KFold so have to use train_test_split
	train, test = train_test_split(range(len(dataset)), train_size=0.2)
	
	#for test, train in kf.split(range(len(dataset))):
	for asdlfkj in range(1):
		print('Training on', train, flush=True)
		train_batch = dataset.collect_batch(train)
		
		X_train = train_batch['terms'].transpose(1, 0, 2, 3, 4)
		print('Training subset: ', X_train.shape)
		X_train = X_train.reshape([-1, *X_train.shape[-4:]])
		X_train = X_train[:, train_batch['mask'].repeat(X_train.shape[-3], axis=-3)].T
		y0_train = train_batch['Fml']
		y0_train = y0_train[train_batch['mask'].repeat(y0_train.shape[-3], axis=-3)][:, None]
		
		input_mask = np.zeros(len(y0_train), dtype=bool)
		input_mask[np.random.randint(len(input_mask), size=train_points)] = True
		
		X_train = X_scaler.fit_transform(X_train[input_mask])
		y0_train = y0_scaler.fit_transform(y0_train[input_mask])
		print('Input, target = ', X_train.shape, y0_train.shape, flush=True)
		
		models = []
		for alpha in tqdm(alphas):
			model = ps.SINDy(optimizer=ElasticNet(alpha=alpha),
							 feature_library=ps.IdentityLibrary())
			model.fit(X_train, x_dot=y0_train)
			models.append(model)

		print('Generated all models, moving to evaluation', flush=True)

		test_info = []
		for i in tqdm(test):
			test_info.append(predict_frame(models, dataset, X_scaler, y0_scaler, i))
		
		test_results = test_results.append(test_info).reset_index(drop=True)
		test_results['cell'] = cell

		print('Finished this train/test split', flush=True)
		test_results.to_csv(f'data/adding_terms/{cell}.csv')

		gc.collect()
	return test_results

from sys import argv
if __name__=='__main__':
	cells = os.listdir('/project/vitelli/cell_stress/ForcePrediction_All_16kpa_new/AM_all_fixed_NormedF_dataset2/testsplit_0')
	cells = [cell for cell in cells if not cell == 'output.out']
	idx = int(argv[1])
	cell = cells[idx]
	test_results = evaluate_cell(cell)
	test_results.to_csv(f'data/adding_terms/{cell}.csv')
