#!/bin/bash

#SBATCH --job-name=automl2
#SBATCH --output=sbatch/%j.automl2.out
#SBATCH --error=sbatch/%j.automl2.err
##SBATCH  --partition=datasci
#SBATCH  --partition=xye

#SBATCH  --gres=gpu:Rtx2080:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=40G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern6 sh run2.sh

