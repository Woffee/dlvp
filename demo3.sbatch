#!/bin/bash

#SBATCH --job-name=gcn3
#SBATCH --output=sbatch/%j.gcn3.out
#SBATCH --error=sbatch/%j.gcn3.err

#SBATCH  --partition=xye

#SBATCH  --gres=gpu:Rtx2080:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=40G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern5 sh run3.sh

