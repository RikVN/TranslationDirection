#!/bin/bash

# Translate all sentence-level files in a given folder
# Do this in both directions, for multiple languages

# First argument is the main folder with the files
fol=$1
# Second argument is the file type. Either euro, macocu or wmt
type=$2

if [[ $type == "euro" || $type == "europarl" ]] ; then
    ext="*sent.tab"
elif [[ $type == "macocu" || $type == "MaCoCU" || $type == "mcc" || $type == "wmt" || $type == "WMT"  ]]; then
    ext="*tab"
else
    echo "Second argument should be either euro, macocu or wmt"
    exit -1
fi

# Main loop
max_length="256"
batch_size="8"
for model in opus nllb m2m; do
    for file in ${fol}/${ext}; do
        base=$(basename $file)
        # Do not translate the "all" file, if it exists
        check_all=$(echo "${base:0:3}")
        if [[ $check_all != "all" ]] ; then
            # Extract src/tgt langs from file names
            if [[ $type == "euro" || $type == "europarl" ]] ; then
                src_lang=$(echo "${base:0:2}")
                tgt_lang=$(echo "${base:3:2}")
            elif [[ $type == "macocu" || $type == "MaCoCU" || $type == "mcc" || $type == "wmt" || $type == "WMT"  ]]; then
                # Files end with .en-${lang}.tab here (or vice versa)
                src_lang=$(echo "${base: -6:2}")
                tgt_lang=$(echo "${base: -9:2}")
            fi

            # Write to specific output file
            out_file="${fol}${base}.${model}"
            echo python src/translate.py --sent_file $file -sl $src_lang -tl $tgt_lang -o $out_file -m $model -tf -ml $max_length -b $batch_size #> ${fol}/${base}.${model}.log
        fi
    done
done
