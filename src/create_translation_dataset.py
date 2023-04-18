#!/usr/bin/env python

'''Create a data set with machine translation from the Europarl files, that we can use for
   fine-tuning LMs in a classification task'''

import sys
import os
import argparse
import random
random.seed(42)
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="Folder with all the Europarl files")
    parser.add_argument("-o", "--out_file", type=str, required=True,
                        help="Output file with the ML data set")
    parser.add_argument("-d", "--data_type", type=str, default="random-mt",
                        choices=["random-mt", "single-mt", "all-mt", "either-single",
                                 "either-random", "all-mt-balanced"],
                        help="How exactly we create the data set. See create_data_set() for \
                              instructions on what each argument means")
    parser.add_argument("-ft", "--format_type", type=str, default="europarl",
                        choices=["euro", "europarl", 'Euro', 'Europarl', "Macocu", "MaCoCu",
                                 "macocu", "wmt", "WMT", "mcc"],
                        help="How are filenames formatted? Differs per type.")
    parser.add_argument("-ti", "--trans_ids", default=["opus", "nllb"], nargs="+",
                        help="IDs of the translation systems we want to incorporate. \
                              Should also be the file extensions of the translations.")
    parser.add_argument("-s", "--single_system", type=str, default="opus",
                        help="MT identifier of the system we use for -d single-mt or either-single")
    args = parser.parse_args()
    return args


def write_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w", encoding="utf8") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def read_sents(in_file):
    '''Read in sentences line by line in UTF-8'''
    return [x.strip() for x in open(in_file, 'r', encoding="utf-8")]


def read_src_tgt(in_file):
    '''Read in first and second tab in a tabbed file'''
    src = []
    tgt = []
    for line in open(in_file, 'r', encoding="utf-8"):
        spl = line.strip().split('\t')
        src.append(spl[0].strip())
        tgt.append(spl[1].strip())
    return src, tgt


def read_data(sent_files, trans_ids, ft):
    '''Read in all the data, e.g. the original texts, HT and MT from the given files'''
    data = []
    for sent_file in sent_files:
        # Src/tgt language identifiers differ per format type
        if ft in ['euro', 'europarl', 'Euro', 'Europarl']:
            # E.g. en-nl.sent.tab
            src_lang = sent_file[0][0:2]
            tgt_lang = sent_file[0][3:5]
        elif ft in ['mcc', 'macocu', 'Macocu', 'MaCoCu']:
            # E.g. dev.hr-en
            src_lang = sent_file[0][-5:-3]
            tgt_lang = sent_file[0][-2:]
        elif ft in ["WMT", "wmt"]:
            # E.g. wmt18.en-tr.tab
            src_lang = sent_file[0][-5:-3]
            tgt_lang = sent_file[0][-2:]
        orig, ht = read_src_tgt(sent_file[1])
        # Read all the translations from the MT files
        mt_list = []
        for mt_file in sent_file[2:]:
            cur_mt = read_sents(mt_file)
            mt_list.append(cur_mt)
            assert len(orig) == len(cur_mt) == len(ht)
        # Flatten the data: keep the identifiers per line
        for idx in range(len(orig)):
            data.append([src_lang, tgt_lang, sent_file[0], sent_file[1], sent_file[2], orig[idx], ht[idx]])
            # Add the MTs here, also add identifier
            for idx2 in range(len(mt_list)):
                data[-1].append(trans_ids[idx2])
                data[-1].append(mt_list[idx2][idx])
    return data


def trans_dict(line):
    '''Create a translation dictionary for convenience'''
    d = {}
    it = iter(line[7:])
    for item in it:
        d[item] = next(it)
    return d


def create_data_set(data, data_type, trans_ids, single_system):
    '''Given an original text, a human translation and multiple machine translations, create a
       pair-wise data set that we can use for ML experiments. One of the two sentences in the pair
       is always original, the other is either machine translated or human translated.

       There are multiple settings possible.
       all-mt   - add all MTs for each instance: 1-orig-ht, 1-orig-opus, 1-orig-nllb, 1-orig-m2m, 2-orig-ht, 2-orig-opus, ..., etc
       all-mt-balanced - add all MTs, same as before, but balance the full data set by duplication (e.g. add the original 3 times if we have 3 MT systems)
       single-mt - add a single MT system for each instance: 1-orig-ht, 1-orig-opus, , 2-orig-ht, 2-orig-opus, etc
                - This is useful for doing cross MT-system evaluation
       random-mt - randomize MT selection for each instance: 1-orig-ht, 1-orig-opus, 2-orig-ht, 2-orig-nllb, 3-orig-ht, 3-orig-nllb, etc
                        - This avoids overfitting on instances - we have enough anyway
       Either HT or MT: for each instance, only add the HT or the MT, instead of both as before.
                      : this has 2 subsettings: taking a single MT system or picking randomly
                      : either-random: 1-orig-opus, 2-orig-ht, 3-orig-ht, 4-orig-nllb, etc
                      : either-single: 1-orig-opus, 2-orig-ht, 3-orig-ht, 4-orig-opus, 5-orig-opus, etc

     In any case we always randomly pick whether original goes first or not, as to not bias the model'''
    out = []
    for line in data:
        # For convenience
        orig = line[5]
        ht = line[6]
        t_d = trans_dict(line)

        # Check the different options
        if data_type == "single-mt":
            out.append([orig, ht, "ht"])
            out.append([orig, t_d[single_system], "mt"])
        elif data_type == "random-mt":
            out.append([orig, ht, "ht"])
            pick = random.choice(trans_ids)
            out.append([orig, t_d[pick], "mt"])
        elif data_type == "all-mt":
            out.append([orig, ht, "ht"])
            for tr_id in trans_ids:
                out.append([orig, t_d[tr_id], "mt"])
        elif data_type == "all-mt-balanced":
            for tr_id in trans_ids:
                out.append([orig, ht, "ht"])
                out.append([orig, t_d[tr_id], "mt"])
        elif data_type in ["either-random", "either-single"]:
            # 50/50 whether we add a machine or human translation
            if random.choice([True, False]):
                # Do HT here, MT for false
                out.append([orig, ht, "ht"])
            elif data_type == "either-random":
                # Pick a random MT system
                pick = random.choice(trans_ids)
                out.append([orig, t_d[pick], "mt"])
            elif data_type == "either-single":
                # For either-single we also pick the same MT system (the single system we specified)
                out.append([orig, t_d[single_system], "mt"])
        else:
            raise ValueError(f"Data type {data_type} is not allowed and/or not implemented")

    # We still always have the original text first, and the translation second
    # There are also no labels yet. Here we turn it into a final string with labels
    # There are four possible labels: first-orig-second-ht, first-orig-second-mt, second-orig-first-ht, second-orig-first-mt
    final = []
    skip = 0
    for vals in out:
        # First we check if the MT, HT or original text is not an empty line
        # Empty lines can occur for the translations sometimes, unfortunately
        # We have no choice but to filter them
        if vals[0].strip() and vals[1].strip():
            if random.choice([True, False]):
                # Switch order of orig here
                label = "second-orig-first-" + vals[2]
                final.append(vals[1] + '\t' + vals[0] + '\t' + label)
            else:
                # Do not switch order of orig here
                label = "first-orig-second-" + vals[2]
                final.append(vals[0] + '\t' + vals[1] + '\t' + label)
        else:
            skip += 1
    print (f"INFO: filtered {skip} instances with an empty translation")
    return final


def main():
    '''Main function'''
    args = create_arg_parser()

    # Get files in folder that are sentence-level per language
    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]

    # Save as [file_name, full_path_name]
    # Europarl, MaCoCu and WMT files have different names, resolve that here
    # Make sure to just read the sentence files and not already translated file (or other types of files)
    if args.format_type in ['euro', 'europarl', 'Euro', 'Europarl']:
        sent_files = [[f, os.path.join(args.folder, f)] for f in files if 'sent' in f and f.endswith('.tab') and not f.startswith('all')]
    elif args.format_type in ['mcc', 'macocu', 'Macocu', 'MaCoCu']:
        sent_files = [[f, os.path.join(args.folder, f)] for f in files if f[-3] == "-" and (f[-2:] == "en" or f[-5:-3] == "en") and not f.startswith('all')]
    elif args.format_type in ["WMT", "wmt"]:
        sent_files = [[f, os.path.join(args.folder, f)] for f in files if f.startswith('wmt') and f.endswith('.tab') and not f.startswith('all')]
    print (f"INFO: working with {len(sent_files)} files")

    # Get corresponding translation files
    for sent_file in sent_files:
        for trans_id in args.trans_ids:
            trans_file = args.folder + '/' + sent_file[0] + '.' + trans_id
            if not os.path.isfile(trans_file):
                raise ValueError(f"File {trans_file} does not exist")
            # Add the translation file
            sent_file.append(trans_file)

    # Read in all the data and add to the list
    data = read_data(sent_files, args.trans_ids, args.format_type)
    print (f"INFO: working with {len(data)} orig-ht pairs")

    # Shuffle this data set so we're completely random
    random.shuffle(data)

    # Now turn it into a ML data set
    final_data = create_data_set(data, args.data_type, args.trans_ids, args.single_system)
    print(f"INFO: created final data set of {len(final_data)} instances")
    write_to_file(final_data, args.out_file)

if __name__ == '__main__':
    # For logging purposes
    print("Generated by command:\npython", " ".join(sys.argv)+'\n')
    main()
