#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --output=sbatch/%j.preprocess.out
#SBATCH --error=sbatch/%j.preprocess.err
##SBATCH  --partition=datasci
#SBATCH  --partition=xye

##SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=64G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern5 sh run_preprocess.sh

