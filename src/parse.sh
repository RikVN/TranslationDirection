#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com
set -eu -o pipefail

# Read in arguments:
# $1: model folder
# $2: sentence file
# $3: LM identifier (e.g. xlm-roberta-base)

echo "Predicting on $2..."

# Make sure to check the order of the labels in src/parse.py
python src/parse.py --model $1 --sent_file $2 --lm_ident $3 > ${2}.log

echo "Done!"

