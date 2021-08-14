#!/bin/bash

#SBATCH --job-name=automl
#SBATCH --output=sbatch/%j.automl.out
#SBATCH --error=sbatch/%j.automl.err
##SBATCH  --partition=datasci
#SBATCH  --partition=xye

#SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=40G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern6 sh run.sh

