#!/bin/bash

#SBATCH --job-name=cf_qemu
#SBATCH --output=cf_qemu.%j.out
#SBATCH --error=cf_qemu.%j.err

##SBATCH  --partition=datasci
#SBATCH  --partition=xye

##SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=32G

module load cuda
module load singularity

singularity exec --nv docker://woffee/keras:cflow sh run.sh

