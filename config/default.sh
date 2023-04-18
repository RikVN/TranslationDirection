#!/bin/bash

# Set data variables
# Either specify here or set in your own config file!
train_file=""
dev_file=""
files_to_parse="v1/en/dev" # add as "file1 file2 file3"
out_prefixes=( dev ) # add as ( dev test eval etc )

# Process variables
limit_train="0" # 0 means no limit
cont="" # use as -c $MODEL_FOLDER to continue training

# Model and training variables
lm_ident="xlm-roberta-base" # You can specify a different one in your own config as well
strategy="epoch"
save_strategy="epoch"
eval_steps="500"
batch_size="32"
patience="1" # Earlystopping patience
learning_rate="1e-5"
weight_decay="0"
max_grad_norm="1"
num_train_epochs="6" # but we use earlystopping with patience 1, so we usually do not get here
warmup_ratio="0.1" # 0.1 best
adafactor="" # add as: --adafactor (did not help in our experiments)
label_smoothing="0.1" # 0.1 best
padding="longest"
grad_check="" # add as --grad_check
eval_accumulation_steps="250"
dropout="0.1"
seed="1234"
ignore_data="" # add as "--ignore_data_skip"
max_length="256"
