#!/bin/bash

#SBATCH --job-name=install_apex
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_install_apex-%j.out

module purge
cd /scratch/mt4493/pretraining/singularity

singularity exec --nv \
      --bind /home/mt4493/resolv.conf:/etc/resolv.conf:ro \
      --overlay /scratch/mt4493/pretraining/singularity/academic_budget_bert.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c 'source /ext3/env.sh;
	                  cd apex;
	                  rm -rf build;
	                  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .'
