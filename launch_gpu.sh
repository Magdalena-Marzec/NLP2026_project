#!/bin/bash
#SBATCH --job-name=NLP2
#SBATCH --partition=common
#SBATCH --qos=dsuwala_common
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00
#SBATCH --output=runs/run2.log

mamba run -n m2026 python src/train.py --data synthetic_data/batch1_v2.tsv --log-file 2_large_r2.log --normalizer-file target_normalizer_large_r2.json

