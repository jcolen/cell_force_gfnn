#!/bin/bash
#
#SBATCH --job-name=sweep_array
#SBATCH --output=/home/jcolen/scratch/sbatch_outputs/%A_%03a.out
#SBATCH --error=/home/jcolen/scratch/sbatch_outputs/%A_%03a.out
#SBATCH --partition=caslake
#SBATCH --array=0-30
#SBATCH --account=pi-vitelli
#SBATCH --ntasks=2
#SBATCH --mem=60G

module load python
conda activate /project/vitelli/ml_venv

cd /home/jcolen/CellProject

for i in 1 2 3 4 5
do
	python sindy_adding_terms.py ${SLURM_ARRAY_TASK_ID}
done
