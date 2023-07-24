import glob
import h5py
import numpy as np
import pandas as pd
import re
import os

from tqdm import tqdm

#Pull dataset into a single h5py file for ease of repeatability/use
#Also, I'm finding h5py kind of nifty right now
data_dir='/project/vitelli/cell_stress/ForcePrediction_All_16kpa_new/AM_all_fixed_NormedF_dataset2/testsplit_0'
cells=['10_cell_3', '08_cell_1', '08_cell_2', '11_cell_1']


with h5py.File('data/cell_dataset.h5', 'w') as h5f:
    dataframe = pd.DataFrame()

    for cell in more_cells:
        filenames = [f for f in glob.glob(os.path.join(data_dir, cell, 'predictions_UNET_c8,4_k3_s2_d1_i6_o2,3', '*.npy'))]
        inds = [list(map(int, re.findall(r'\d+', f)))[-1] for f in filenames]
        files = [f for _, f in sorted(zip(inds, filenames))]

        di = pd.DataFrame({
            'cell': [cell,]*len(files),
            'cell_idx': np.arange(len(files)),
        })
        if not cell in h5f:
            h5f.create_group(cell)
            
        for i in tqdm(range(len(files))):
            cell_idx = '%03d' % (di.cell_idx[i])
            if not cell_idx in h5f[cell].keys():
                h5f[cell].create_group(cell_idx)

            fml = np.load(files[i], mmap_mode='r')[0]
            zyx = np.load(files[i].replace('predictions', 'inputs'), mmap_mode='r')[0]
            fex = np.load(files[i].replace('predictions', 'outputs'), mmap_mode='r')[0]
            g = h5f[cell+'/' + cell_idx]
            fml = fml[0] * np.stack([
                np.cos(fml[1]),
                np.sin(fml[1]),
            ])
            g.create_dataset('F_exp', data=fex)
            g.create_dataset('F_ml', data=fml)
            g.create_dataset('zyxin', data=zyx)
            g.create_dataset('mask', data=zyx!=0)
        
        dataframe = dataframe.append(di).reset_index(drop=True)
                         
dataframe.to_csv('data/cell_dataset.csv', index=False)
