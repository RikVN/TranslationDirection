# Preprocessing

This is the annotated version of our preprocessing for the data used in translation direction experiments.

You can either follow those instructions step by step, or just simply run:

```
./src/data_preprocessing.sh
```

We will process the Europarl data to sentence-level and document-level. For document-level, **note** that we set a max length so it still fits in the model size of the LM we use. In our case XLM-R, which has at most 512 tokens. Giving some room for special tokens, we set a max of 500 tokens per document.

```
mkdir -p exp_data
mkdir -p exp_data/europarl/
mkdir -p exp_data/europarl/full/
for fol in europarl-extract/corpora/parallel/*/; do
    pair=$(basename $fol)
    python src/get_documents.py -i ${fol}/tab/ -o exp_data/europarl/full/${pair}.sent.tab -m 0 -id ${pair}
    python src/get_documents.py -i ${fol}/tab/ -o exp_data/europarl/full/${pair}.doc.tab -m 500 -id ${pair}
done
```

Now in ``exp_data/europarl/full/`` we have files such as ``bg-en.doc.tab`` with the Bulgarian original data and English translations at the document-level. There is also ``en-bg.doc.tab`` with the opposite translation direction.


There is an issue: there is a lot more original English data for any pair that includes English. Similarly, some language pairs in general have a lot more data than other language-pairs. If we want to train a multi-lingual system, we probably want equal amounts of data for each language pair (as much as possible).

So, we want to down-sample the data sets. First, we shuffle the original data sets and save them, so we always go back. **Note** that this way, we lose the order of the sentences/documents, but this happens anyway during training of system.

We also want to deduplicate this data, as there are likely a lot of similar sentences (EU boilerplate). We use ``src/dedup.py`` to do this, which filters all near-duplicate sentences and documents.

```
mkdir exp_data/europarl/full_shuf_dedup/
cd exp_data/europarl/full/
for file in *tab ; do
    cat $file | python ../../../src/dedup.py | shuf > ../full_shuf_dedup/${file}
done
cd ../../../
```

Before actually down-sampling, we have to think about dev and test sets. We have different test sets available from WMT, and also ones we selected ourselves from the MaCoCu project. But we also want to split off dev/test for each Europarl test.

This also depends on which language pair you're interested in. In our case, we're interested in Bulgarian, so we take a larger dev/test part for this language. For the other languages, we keep 250 instances per file for the sentence-level, and 100 for the document-level.

Just run the ``src/split_dev_test.sh`` script to take care of this:

```
mkdir exp_data/europarl/split
./src/split_dev_test.sh exp_data/europarl/full_shuf_dedup/ exp_data/europarl/split/
```

Now that we have this, do the actual down-sampling on the training sets of each file and save them all to specific down-sampling folders:

```
mkdir -p exp_data/europarl/split/train/down/
cd exp_data/europarl/split/train/
for size in 100 250 500 1000 2000 3000 5000 10000; do
    mkdir -p down/${size}/
    for file in *tab; do
        head -${size} $file > down/${size}/${file}
    done
done
```

Note that **not** all files actually have the amount of lines specified, e.g. if a file has 7000 lines and we do downsample to 10000, we just keep the 7000 as is.

Still, these are not training sets, as they're all individual sets in one direction. Let's create training sets from all down-sampled sets.

There's a problem though: in these data sets the first text is always the original. It's also still in order per language. If we are training a model, we want to randomize the first and second sentence so that there is no bias introduced in that regard.

Therefore, we shuffle the training set and randomize whether the first or second sentence was the original. We also remove the meta data for these files, so it can used directly for training:

```
for size in 100 250 500 1000 2000 3000 5000 10000; do
    for type in sent doc ; do
        cat down/${size}/*en*${type}.tab > down/${size}/all.${type}.tab
        shuf down/${size}/all.${type}.tab > down/${size}/all.${type}.tab.shuf
        cat down/${size}/all.${type}.tab.shuf | python ../../../../src/randomize_order.py > down/${size}/all.${type}.tab.shuf.frm
    done
done
cd ../../../../
```

We also want to download the WMT test sets we want to work with. First get Turkish from WMT16, WMT17 and WMT18:

```
mkdir -p exp_data/wmt/
for typ in wmt16 wmt17 wmt18; do
    sacrebleu -t ${typ} -l en-tr --echo src ref origlang > tmp
    awk -F '\t' '$3=="en"' tmp | cut -f1,2 > exp_data/wmt/${typ}.en-tr.tab
    awk -F '\t' '$3=="tr"' tmp |  awk -F '\t' '{printf("%s\t%s\n", $2, $1)}' > exp_data/wmt/${typ}.tr-en.tab
done
rm tmp
```

And Icelandic from WMT21 (this conference already separated the data by original language):

```
sacrebleu -t wmt21 -l en-is --echo src ref > exp_data/wmt/wmt21.en-is.tab
sacrebleu -t wmt21 -l is-en --echo src ref > exp_data/wmt/wmt21.is-en.tab
```

Similarly as we before, we shuffle the data and convert it to a format the model can work with:

```
cd exp_data/wmt/
for typ in wmt16 wmt17 wmt18; do
    cat ${typ}.tr-en.tab ${typ}.en-tr.tab | shuf > ${typ}_both_tr_en.tab.shuf
    cat ${typ}_both_tr_en.tab.shuf | python ../../src/randomize_order.py > ${typ}_both_tr_en.tab.shuf.frm
done
cat wmt21.is-en.tab wmt21.en-is.tab | shuf > wmt21_both_is_en.tab.shuf
cat wmt21_both_is_en.tab.shuf | python ../../src/randomize_order.py > wmt21_both_is_en.tab.shuf.frm
cd ../../
```

There is also data for Croatian and Slovene from the MaCoCu release, that we annotated ourselves. Please download the train, dev and test split like this:

```
cd exp_data
wget https://www.let.rug.nl/rikvannoord/TD/macocu.zip
unzip macocu.zip
```

Put the sets in the correct format for training and evaluation:

```
cd macocu/
for lang in sl hr; do
    for type in train dev test ; do
        cat ${lang}/${type}.${lang}-en ${lang}/${type}.en-${lang} | shuf > ${lang}/${type}.both.en.${lang}.shuf
        cat ${lang}/${type}.both.en.${lang}.shuf | python ../../src/randomize_order.py > ${lang}/${type}.both.en.${lang}.shuf.frm
    done
done
```

This data is already shuffled. We also create down-sampled training splits again, which we put in the right format:

```
for lang in sl hr; do
    cd ${lang}
    mkdir -p down
    for size in 5000 10000 20000 30000 50000; do
        mkdir -p $size
        head -${size} train.en-${lang} > down/${size}/train.en-${lang}
        head -${size} train.${lang}-en > down/${size}/train.${lang}-en
        cat down/${size}/train.en-${lang} down/${size}/train.${lang}-en | shuf > down/${size}/train.both.en.${lang}.shuf
        cat down/${size}/train.both.en.${lang}.shuf |  python ../../../src/randomize_order.py > down/${size}/train.both.en.${lang}.shuf.frm
    done
    cd ../
done
```

Now preprocessing is complete! For training and parsing, go back to the main [README](README.md).
