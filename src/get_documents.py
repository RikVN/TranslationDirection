#!/usr/bin/env python
# -*- coding: utf8 -*-

'''Save the individual files of Europarl as parallel documents as much as possible, as long as they
   do not exceed the max length of 500 (for both). Add as tab-separated file that also includes
   the name of the original file so we can always go back'''

import sys
import argparse
import os
from transformers import AutoTokenizer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, type=str,
                        help="Folder in which we find all the files")
    parser.add_argument("-o", "--output_file", required=True, type=str,
                        help="Output file to write to")
    parser.add_argument("-t", "--tokenizer_id", default="xlm-roberta-large", type=str,
                        help="Lang model ID of the tokenizer we will use")
    parser.add_argument("-m", "--max_len", default=500, type=int,
                        help="Max length of a document - 0 means do sentence level")
    parser.add_argument("-id", "--ident", required=True, type=str,
                        help="ID of lang pair, e.g. en-bg")
    args = parser.parse_args()
    return args


def write_to_file(lst, out_file, do_strip=True):
    '''Write list to file'''
    with open(out_file, "w", encoding="utf8") as out_f:
        for line in lst:
            if do_strip:
                out_f.write(line.strip() + '\n')
            else:
                out_f.write(line + '\n')
    out_f.close()


def chunk_add(docs, src_sents, tgt_sents, tokenizer, max_len, in_file, ident):
    '''Add documents in chunks of sentences that never exceed max_len'''
    src_keep, tgt_keep = "", ""
    cur_len, snt_count = 0, 0

    for src, tgt in zip(src_sents, tgt_sents):
        # Max length of 0 just means do sentence-level
        if max_len == 0:
            docs.append([src.strip(), tgt.strip(), "1", in_file, ident])
        else:
            # Get the length of the tokenized sentences
            # Since we feed both to the classifier later, we also add both to the total
            len_src = len(tokenizer(src.strip())["input_ids"])
            len_tgt = len(tokenizer(tgt.strip())["input_ids"])
            # To potentially check
            #tok_src = tokenizer(src.strip())["input_ids"]
            #tok_tgt = tokenizer(tgt.strip())["input_ids"]
            if (cur_len + len_src + len_tgt) > max_len:
                # Document would grow too big, add now already
                # Only add if we already have content, i.e. ignore single sentences
                # that are larger than the max length (should be extremely rare but can happen)
                if src_keep and snt_count > 0:
                    # Output:src_text, tgt_text, sentence_count, length_in_tokens, specific_file_name, lang identifier
                    docs.append([src_keep, tgt_keep, str(snt_count), str(cur_len), in_file, ident])
                    src_keep = src.strip()
                    tgt_keep = tgt.strip()
                    cur_len = len_src + len_tgt
                    # Reset sentence count and avg score
                    snt_count = 0
            else:
                # Just add sentence, they are not too large yet
                src_keep += ' ' + src.strip()
                tgt_keep += ' ' + tgt.strip()
                cur_len += (len_src + len_tgt)
            snt_count += 1

    # Eventually, we also add if we never exceed the max len of course
    if src_keep:
        docs.append([src_keep, tgt_keep, str(snt_count), str(cur_len), in_file, ident])
    return docs


def main():
    '''Main function for restoring paragraphs/documents from macocu data'''
    args = create_arg_parser()

    # First get all files in the input folder and order them based on the date
    fls = []
    for root, dirs, files in os.walk(args.input_folder):
        for f in files:
            add_f = int(f[0:2])
            # Use a hacky method to put 90-99 in front of 01/02 etc
            if f.startswith('9'):
                add_f = -1
            fls.append([add_f, f, os.path.join(root, f)])

    # Now just sort by date
    sorted_files = [y[1:3] for y in sorted(fls, key=lambda x: (x[0], x[1]))]
    print(f"Found {len(sorted_files)} files for {args.ident}")

    # Already set the tokenizer here
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # Loop over the files and get the documents out
    docs = []
    for s in sorted_files:
        in_file, full_file = s
        lines = [x.strip().split('\t') for x in open(full_file, 'r', encoding="utf8")]
        src_sents = [x[0] for x in lines]
        tgt_sents = [x[1] for x in lines]
        docs = chunk_add(docs, src_sents, tgt_sents, tokenizer, args.max_len, in_file, args.ident)
    print(f"Found {len(docs)} docs for {args.ident}")
    # Write to tab-separated file
    write_to_file(["\t".join(d) for d in docs], args.output_file)

if __name__ == '__main__':
    main()
