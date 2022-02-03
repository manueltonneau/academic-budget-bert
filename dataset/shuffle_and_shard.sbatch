#!/bin/bash

#SBATCH --job-name=shuffle_and_shard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=600GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_shuffle_and_shard_%j.out

INPUT_TXT=$1
OUTPUT_FOLDER=$2

mkdir -p /scratch/mt4493/pretraining/data/shards/${OUTPUT_FOLDER}


module purge
#module load anaconda3/2020.02
#source /scratch/mt4493/twitter_labor/code/envs/preprocessing_env/bin/activate

shuf -o /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/shuffled.txt /scratch/mt4493/pretraining/data/txts/${INPUT_TXT}
echo 'shuffled the input txt file'
line_count=$(sed -n '$=' /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/shuffled.txt)
train_len=$(($((line_count * 9)) / 10))
test_len=$((${line_count} - ${train_len}))
csplit /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/shuffled.txt ${train_len}
echo 'splitted the shuffled file into two'

split -a 5 -d -l $((${line_split} / 256)) --additional-suffix=.txt /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/xx00 /scratch/mt4493/pretraining/data/shards/${OUTPUT_FOLDER}/training
echo 'created train shards'
split -a 5 -d -l $((${line_split} / 128)) --additional-suffix=.txt /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/xx01 /scratch/mt4493/pretraining/data/shards/${OUTPUT_FOLDER}/test
echo 'created test shards'

rm /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/xx00
rm /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/xx01
rm /scratch/mt4493/pretraining/data/txts/${OUTPUT_FOLDER}/shuffled.txt
echo 'removed useless txt files'