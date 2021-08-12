#!/bin/bash

#SBATCH --job-name=t_pre
#SBATCH --output=sbatch/%j.t_pre.out
#SBATCH --error=sbatch/%j.t_pre.err

##SBATCH  --partition=datasci
#SBATCH  --partition=xye

##SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=64G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern5 sh run_treecaps.sh

