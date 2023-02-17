#!/usr/bin/env python
# -*- coding: utf8 -*-

# Randomly change order of first or second sentence that is original
# Remove other information other than the sentences
# Add the label

import sys
from random import choice
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Read from std in
for idx, line in enumerate(sys.stdin):
    # Get both sentences, ignore everything else
    sent1 = line.strip().split('\t')[0].strip()
    sent2 = line.strip().split('\t')[1].strip()
    # Do 50/50 original first or translation first
    # Except for the first item, so that the labels are always in the correct order
    if choice([True, False]) or idx == 0:
        fst = sent1
        snd = sent2
        label = "first-orig"
    else:
        label = "second-orig"
        fst = sent2
        snd = sent1
    # Print output to std out
    print(f"{fst}\t{snd}\t{label}")
