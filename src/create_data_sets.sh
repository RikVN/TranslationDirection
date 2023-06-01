#!/bin/bash
# Create training/evaluation data sets from a given input file and translation files
set -eu -o pipefail

# First argument is the input folder we create data sets from
folder=$1
# Second argument is the file identifier: either euro, macocu or wmt
ident=$2

# Files with translations such as ${file}.opus should exist. If not, create them using ./src/translate.sh
# You can also change the arguments here, e.g. if you only have opus and nllb just remove m2m
mt_systems="opus nllb"

# Now create data sets for all the different settings
# If you don't care about all settings, just remove them
settings="all-mt all-mt-balanced single-mt random-mt either-random either-single"

# Main loop
for setting in $settings; do
    # For the two "single" settings, create data sets for all MT systems
    if [[ $setting == "single-mt" || $setting == "either-single" ]]; then
        for mt_sys in $mt_systems; do
            # Create single-mt-opus files, etc
            python src/create_translation_dataset.py -f $folder -o ${folder}${setting}-${mt_sys}.all.clf -d $setting -ft $ident -ti $mt_systems -s $mt_sys
            python src/order_label.py ${folder}${setting}-${mt_sys}.all.clf
        done
    else
        # Just a single run for this setting
        python src/create_translation_dataset.py -f $folder -o ${folder}${setting}.all.clf -d $setting -ft $ident -ti $mt_systems
        python src/order_label.py ${folder}${setting}.all.clf
    fi
done
