import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from pathlib import Path

def run_validation_epoch(model,
					     loss_func,
						 device,
					     val_loader):
	base_loss = 0.
	kern_loss = 0.
	with torch.no_grad():
		for batch in val_loader:
			x, y0 = batch
			
			y = model.forward(x.to(device))
			loss = loss_func(y, y0.to(device))
			
			kernel = model.get_kernel()
			kernel_mag = (torch.conj(kernel) * kernel).real.abs()
			kernel_loss = kernel_mag.sum() - kernel_mag[..., 0, 0].sum()
			
			kern_loss += kernel_loss.item() / len(val_loader)
			base_loss += loss.item() / len(val_loader)

	return base_loss, kern_loss

def run_training_epoch(model,
					   optimizer,
					   loss_func,
					   beta,
					   device,
					   train_loader,
					   accumulate_batch=1,
					   clip=None):
	i = 0
	for batch in train_loader:
		if i % accumulate_batch == 0:
			optimizer.zero_grad()

		x, y0 = batch
		
		y = model.forward(x.to(device))
		loss = loss_func(y, y0.to(device))
		
		kernel = model.get_kernel()
		kernel_mag = (torch.conj(kernel) * kernel).real.abs()
		kernel_loss = kernel_mag.sum() - kernel_mag[..., 0, 0].sum()

		loss = loss + beta * kernel_loss
		loss.backward()
		if i % accumulate_batch == 0:
			if clip is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()

		i += 1

def train_loop(model,
			  train_loader,
			  val_loader,
			  hparams,
			  loss_func,
			  save_path,
			  epochs=100,
			  accumulate_batch=1,
			  scheduler=None,
			  clip=None):

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	optimizer = torch.optim.Adam([
		{'params': model.conv_in.parameters(), 'lr':  hparams['base_lr']},
		{'params': model.kernel, 'lr':  hparams['kernel_lr']},
	])

	if scheduler is not None:
		scheduler = scheduler(optimizer, patience=20)

	save_dir = Path(save_path).parent.absolute()
	print(save_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	best_loss = 1e10
	for epoch in range(epochs):
		run_training_epoch(
			model, 
			optimizer,
			loss_func, 
			hparams['beta'], 
			device, 
			train_loader,
			accumulate_batch=accumulate_batch,
			clip=clip)
		base_loss, kern_loss = run_validation_epoch(
			model,
			loss_func,
			device,
			val_loader)

		outstr = 'Epoch=%d\tBase Loss=%g\tKernel Loss=%g' % (epoch, base_loss, kern_loss)
		print(outstr)

		if scheduler is not None:
			if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				loss = base_loss + hparams['beta'] * kern_loss
				scheduler.step(loss)
			else:
				scheduler.step()

		if base_loss < best_loss:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': hparams,
				'epoch': epoch,
				'loss': base_loss}

			torch.save(save_dict, save_path)
			best_loss = base_loss

from argparse import ArgumentParser
from gfnn_models import *
from gfnn_data_processing import *

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--base_lr', type=float, default=1e-4)
	parser.add_argument('--kernel_lr', type=float, default=1e-2)
	parser.add_argument('--beta', type=float, default=1e-1)
	parser.add_argument('--cell_crop', type=int, default=1024),
	parser.add_argument('--nmax', type=int, default=50),
	parser.add_argument('--downsample', type=int, default=4)
	parser.add_argument('--scheduler', action='store_true')
	parser.add_argument('--real', action='store_true')
	parser.add_argument('--coulomb', action='store_true')
	parser.add_argument('--epochs', type=int, default=200)
	hparams = vars(parser.parse_args())
	size = hparams['cell_crop'] // hparams['downsample']
	hparams['size'] = (size, size)

	dataset = Dataset()
	
	dl_kwargs = dict(
		num_workers=4,
		batch_size=8,
		shuffle=True,
		pin_memory=True)

	washout_split = 120 #Don't train on last cell (washout)
	train = Subset(dataset, np.arange(len(dataset) - washout_split))
	val = Subset(dataset, np.arange(len(dataset)-washout_split, len(dataset)))
	train_loader = DataLoader(train, **dl_kwargs)
	val_loader = DataLoader(val, **dl_kwargs)

	if hparams['real'] or hparams['coulomb']:
		hparams['size'] = hparams['size'][0]
		if hparams['coulomb']:
			model = RealCoulombGFNN(**hparams)
		else:
			model = RealGFNN(**hparams)
	else:
		model = ClebschGFNN(**hparams)

	dataset.transform = model.get_transform(**hparams)

	save_path = os.path.join(
		'/home/jcolen/CellProject/data/tb_logs',
		'UnseenWashout_%s_down=%d_beta=%g' % (model.__class__.__name__, hparams['downsample'], hparams['beta']))
	print(model.__class__.__name__)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau if hparams['scheduler'] else None
			
	loss_func = F.mse_loss

	train_loop(
		model,
		train_loader,
		val_loader,
		hparams,
		loss_func,
		save_path,
		scheduler=scheduler,
		epochs=hparams['epochs'],
		clip=0.5)
