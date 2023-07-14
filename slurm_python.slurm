#!/bin/bash
#
#SBATCH --job-name=train_CellMechanics
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/Cell_Mechanics_%A_AA_BB.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/Cell_Mechanics_%A_AA_BB.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --ntasks=2

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/CellProject

epochs=200
downsample=4
cell_crop=1024

beta=BB
echo "Launching beta = ${beta}"
python training.py \
	--epochs $epochs \
	--cell_crop $cell_crop \
	--downsample $downsample \
	--beta $beta \
	--base_lr 1e-4 --kernel_lr 1e-2 \
	--real
