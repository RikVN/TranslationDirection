#!/bin/bash

# Set data variables
# Either specify here or set in your own config file
train_file="exp_data/europarl/split/train/down/3000/random-mt.all.clf.ord"
dev_file="exp_data/europarl/split/dev/random-mt.all.clf.ord"
dev_macocu_sl="exp_data/macocu/sl/dev/random-mt.all.clf.ord"
wmt16_tr="exp_data/wmt/wmt16/random-mt.all.clf.ord"
files_to_parse="$dev_file $dev_macocu_sl $wmt16_tr"
out_prefixes=( dev_euro dev_macocu_sl wmt16_tr )
num_train_epochs="1"
