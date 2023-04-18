#!/usr/bin/env python

'''Simple script that shuffle the data, but makes sure the first 4 instances have the exact label
   order we want. Just to make absolutely sure nothing weird is going on there.'''

import sys
from random import shuffle
from create_translation_dataset import write_to_file

# Read the lines
lines = [x.strip() for x in open(sys.argv[1], 'r', encoding="utf-8")]

# Initial shuffle so we always shuffle the full file
shuffle(lines)
sample = lines[0:500]
rest = lines[500:]
labels = ["first-orig-second-ht", "first-orig-second-mt",
          "second-orig-first-ht", "second-orig-first-mt"]

success = False
# Shuffle randomly until we get the right order.
# There are nicer solutions but this is fast enough for 4 labels
while not success:
    shuffle(sample)
    labs = [x.split('\t')[2] for x in sample][0:4]
    if labs == labels:
        success = True
        break

full = sample + rest
write_to_file(full, sys.argv[1] + '.ord')
