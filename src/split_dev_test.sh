#!/bin/bash
# Script that splits off dev/test from all Europarl files
# $1: input folder with all files (.tab extension)
# $2: main folder in which we save train/dev/test splits

set -eu -o pipefail

function split_data(){
    # Function to split dev/test off an input file
    # $1: input file
    # $2: size of dev and test set (individual!), e.g. if you specify 200
    #     the size of dev is 200 and the size of test is 200
    # #3: the main folder in which we save the splits

    name=$(basename $1)
    # Calculate full size of file
    lines=$(cat $1 | wc -l)
    let double=$(( ${2} * 2 ))

    # Split of dev + test
    head -${double} $1 > tmp
    # Save dev and test to individual files
    head -${2} tmp > ${3}/dev/${name}
    tail -${2} tmp > ${3}/test/${name}
    # Calculate how much we keep of original file
    let keep=$(( ${lines} - ${double} ))
    # Overwrite original file with dev/test split off
    tail -${keep} ${1} > ${3}/train/${name}
    # Cleaning and printing
    rm tmp
    echo "Created ${3}/train/${name} of $keep lines"
    echo "Created ${3}/dev/${name} of $2 lines"
    echo "Created ${3}/test/${name} of $2 lines"
    echo
}

# Create train/dev/test subfolders in folder $2
mkdir -p ${2}/train ${2}/dev ${2}/test

# Loop over files in folder
for file in ${1}/*tab; do
    base=$(basename $file)
    first_lang="${base:0:2}"
    sec_lang="${base:3:5}"
    # For Bulgarian we have a different size split, since we have that
    # language in MaCoCu
    if [[ $first_lang = "bg" || $sec_lang = "bg" ]] ; then
        sent_size="500"
        doc_size="200"
    else
        sent_size="250"
        doc_size="100"
    fi

    # Check if we're dealing with sentences or documents
    if [[ "$base" == *"sent"* ]]; then
        size=$sent_size
    else
        size=$doc_size
    fi

    # Now do the actual splitting, and save in train/ dev/ and test/ folders
    split_data $file $size $2
done
