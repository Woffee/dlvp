#!/bin/bash

#SBATCH --job-name=gcn2
#SBATCH --output=sbatch/%j.gcn2.out
#SBATCH --error=sbatch/%j.gcn2.err

##SBATCH  --partition=datasci
#SBATCH  --partition=xye

#SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=48G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern5 sh run2.sh

