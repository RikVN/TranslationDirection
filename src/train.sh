#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=1-23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com
set -eu -o pipefail

# We read the settings from two files: a file with default settings
# and a file with our specific experimental settings (override) that is the argument in $1
source config/default.sh
source $1

# Set main directory: in our experiments this always works,
# configs are in config/, experiments in exps/
base_dir=$(dirname "$1")
main_dir=$(echo ${base_dir}/ | sed 's/config/exps/')
filename=$(basename -- "$1")
filename="${filename%.*}"
out_fol="$main_dir${filename}"

# Setup folder structure
for nme in log models output eval bkp; do
    mkdir -p ${out_fol}/${nme}
done

# Training call with all the settings (either from config/default.sh or our own config)
python src/train.py --train_file $train_file --dev_file $dev_file -lm $lm_ident -l $limit_train $cont -o ${out_fol}/models/ -str $strategy -ss $save_strategy --eval_steps $eval_steps -bs $batch_size --patience $patience -lr $learning_rate -wd $weight_decay -mgn $max_grad_norm -ne $num_train_epochs -wr $warmup_ratio $adafactor -ls $label_smoothing -of ${out_fol}/output/ -pa $padding $grad_check -eas $eval_accumulation_steps --max_length $max_length --dropout $dropout --seed $seed $ignore_data > ${out_fol}/log/train.log

# Do parsing for specified dev/test files, save eval files
count=0
for test_file in $files_to_parse; do
    echo "Producing output for $test_file"
    echo "Writing to ${out_fol}/output/${out_prefixes[$count]}"
    # We save only 1 checkpoint so we can select it like this
    python src/parse.py -m ${out_fol}/models/ --train_log ${out_fol}/log/train.log --lm_ident $lm_ident --sent_file $test_file -o ${out_fol}/output/${out_prefixes[$count]} -pa $padding -ml $max_length -cm ${out_fol}/eval/cm_${out_prefixes[$count]}.pdf > ${out_fol}/eval/${out_prefixes[$count]}.eval
    (( count++ )) || true
done

# Backup for reproducibility: the config and default files, just to be sure
cp config/default.sh ${out_fol}/bkp/
cp $1 ${out_fol}/bkp/


