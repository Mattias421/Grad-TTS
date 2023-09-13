#!/bin/bash
#SBATCH --output=./slurm/%j.out
module load Anaconda3/2022.10
source activate grad-tts

for i in $(seq $((SLURM_ARRAY_TASK_ID*100)) $((SLURM_ARRAY_TASK_ID * 100 + 100))); do
    python get_score_parallel.py --config-name=generate_scores_parallel n_best_dataset_index=$i
done