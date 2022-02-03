#!/bin/bash

#SBATCH --job-name=generate_samples
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_generate_samples_%j.out

INPUT_FOLDER=$1

mkdir -p /scratch/mt4493/pretraining/data/pretraining_files/${INPUT_FOLDER}

module purge
#module load anaconda3/2020.02
#source /scratch/mt4493/twitter_labor/code/envs/preprocessing_env/bin/activate

cd /scratch/mt4493/pretraining/code/academic-budget-bert/dataset

singularity exec --overlay /scratch/mt4493/pretraining/singularity/academic_budget_bert.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 generate_samples.py \
    --dir /scratch/mt4493/pretraining/data/shards/${INPUT_FOLDER} \
    -o /scratch/mt4493/pretraining/data/pretraining_files/${INPUT_FOLDER} \
    --dup_factor 10 \
    --seed 42 \
    --vocab_file /scratch/mt4493/pretraining/init/roberta.base/vocab.json \
    --do_lower_case 0 \
    --masked_lm_prob 0.15 \
    --max_seq_length 128 \
    --max_predictions_per_seq 20 \
    --n_processes 1 \
    --model_name roberta-base"