#!/bin/bash

#SBATCH --job-name=shard_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_gradual_preprocessing_%j.out

INPUT_FOLDER=$1

mkdir -p /scratch/mt4493/pretraining/data/pretraining_files/${INPUT_FOLDER}

module purge
#module load anaconda3/2020.02
#source /scratch/mt4493/twitter_labor/code/envs/preprocessing_env/bin/activate

cd /scratch/mt4493/pretraining/code/academic-budget-bert/dataset

singularity exec --overlay /scratch/mt4493/pretraining/singularity/academic_budget_bert.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 shard_data.py \
	        --dir /scratch/mt4493/pretraining/data/txts/${INPUT_FOLDER} \
          -o /scratch/mt4493/pretraining/data/pretraining_files/${INPUT_FOLDER} \
          --num_train_shards 256 \
          --num_test_shards 128 \
          --frac_test 0.1"