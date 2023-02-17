#!/bin/bash

# Set data variables
# Either specify here or set in your own config file
train_file="exp_data/europarl/split/train/down/1000/all.sent.tab.shuf.frm"
dev_file="exp_data/europarl/split/dev/both.bg.en.sent.shuf.frm"
files_to_parse="${dev_file} exp_data/macocu/hr/dev.both.en.hr.shuf.frm exp_data/macocu/sl/dev.both.en.sl.shuf.frm exp_data/wmt/wmt16_both_tr_en.tab.shuf.frm"
out_prefixes=( dev_euro_bg dev_macocu_hr dev_macocu_sl wmt16_tr)
lm_ident="xlm-roberta-base"
num_train_epochs="3"
