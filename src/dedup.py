from unicodedata import category as cat
from unidecode import unidecode
import sys

# Translate table to remove non alphabetic characters
tbl = [chr(i) for i in range(sys.maxunicode) if not cat(chr(i)).startswith('L')]
remove_non_alpha = str.maketrans('', '', ''.join(tbl))

# Remove near duplicates, only print ones that were not duplicates
d1= {}
d2 = {}
for line in sys.stdin:
    # If *either* sentence 1 or sentence 2 is already in the data, we filter this pair
    sent1 = line.strip("\n").lower().split('\t')[0].strip().translate(remove_non_alpha)
    sent2 = line.strip("\n").lower().split('\t')[1].strip().translate(remove_non_alpha)
    if sent1 not in d1 and sent2 not in d2:
        print(line.strip())
        d1[sent1] = 1
        d2[sent2] = 1
