#!/bin/bash

#SBATCH --job-name=pretraining_roberta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=167:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_pretraining_roberta-%j.out

DATA_FOLDER=$1

module purge

DATA_DIRECTORY=/scratch/mt4493/pretraining/data/pretraining_files/${DATA_FOLDER}

#VOCAB_SIZE_DIR=${DATA_DIRECTORY}/${DATA_FOLDER}/${DATA_FOLDER}_bpe/${VOCAB_SIZE}
#DATA_DIR=${VOCAB_SIZE_DIR}/pretraining_files
#pip install --editable /usr/share/torch-xla-1.7/tpu-examples/deps/fairseq

#TOTAL_UPDATES=1000000 # Total number of training steps
#WARMUP_UPDATES=10000  # Warmup the learning rate over this many updates
#PEAK_LR=0.0005        # Peak learning rate, adjust as needed
#if [ ${INIT} = "roberta_base" ]; then
#  TOKENS_PER_SAMPLE=512 # Max sequence length
#else
#    TOKENS_PER_SAMPLE=128 # Max sequence length
#fi
#MAX_SENTENCES=8       # Number of sequences per batch (batch size)
#UPDATE_FREQ=32       # Increase the batch size 16x
#
#if [ ! -d ${DATA_DIR}/tensorboard ]; then
#  mkdir -p ${DATA_DIR}/tensorboard
#fi


singularity exec --nv \
      --bind /home/mt4493/resolv.conf:/etc/resolv.conf:ro \
      --overlay /scratch/mt4493/pretraining/singularity/academic_budget_bert.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; deepspeed run_pretraining.py \
  --model_type bert-mlm-roberta --tokenizer_name roberta-base \
  --load_training_checkpoint /scratch/mt4493/pretraining/data/init/roberta.base.tar.gz \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule time \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 24.0 \
  --early_exit_time_marker 24.0 \
  --dataset_path ${DATA_DIRECTORY} \
  --output_dir /scratch/mt4493/pretraining/data/checkpoints \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name pretraining_experiment \
  --project_name budget-roberta-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 180 \
  --early_stop_eval_loss 6 \
  --seed 42 \
  --fp16"