#!/bin/bash

#SBATCH --job-name=cf_gcn
#SBATCH --output=cf_gcn.%j.out
#SBATCH --error=cf_gcn.%j.err

##SBATCH  --partition=datasci
#SBATCH  --partition=xye

##SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=32G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern5 sh run.sh

