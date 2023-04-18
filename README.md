# TranslationDirection

Code for experiments regarding determining translation direction of parallel sentences.

## Getting started

Setup a Conda environment:

```
conda create -n trdi
conda activate trdi
```

Clone the repo and install the requirements:

```
git clone https://github.com/RikVN/TranslationDirection
cd TranslationDirection
pip install -r requirements.txt
```

**Note:** if you want to use the previous code for just translation direction classification, revert the repo like this and follow the previous README instead.

```
git checkout 192efc1
```

## Data ##

### Europarl Extract Preprocessing

The following steps are added for completeness, but will take quite a long time. 

**You can also immediately to [Our preprocessing](#our-preprocessing) below and just download the preprocessed files.**

If you want to do your own preprocessing from scratch, take the following steps:

We have to extract and preprocess the Europarl data we will use for training. We will first use the [europarl-extract](https://github.com/mustaszewski/europarl-extract) repo to download and clean the data:

```
git clone https://github.com/mustaszewski/europarl-extract
cd europarl-extract
```

Get the scripts and data (downloading and extracting takes some time):

```
wget https://github.com/mustaszewski/europarl-extract/archive/v0.9.tar.gz
tar xzf v0.9.tar.gz
cd europarl-extract-0.9
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar xzf europarl.tgz
```

Now we have to preprocess the source files for removing markup, empty lines, etc. This process takes a long time unfortunately:

```
./preprocess/cleanSourceFiles.sh txt/
```

Ensure that no two statements are assigned the same ID in a source file:

```
python disambiguate_speaker_IDs.py txt/
```

Now do sentence segmentation:

**Note that there is a mistake in the original script**. Open ``preprocess/segment_EuroParl.sh`` and remove ``europarl_extract/`` in line 24, so the filepath is correct. Then run:

```
./preprocess/segment_EuroParl.sh txt/
```

Finally, extract the parallel corpora and get both directions:

```
cd ../
mkdir -p corpora
export s_file="corpora/europarl_statements.csv"
for fol in europarl-extract-0.9/txt/*/; do
    lang=$(basename $fol)
    python extract.py parallel -sl EN -tl ${lang^^} -i europarl-extract-0.9/txt/ -o corpora/ -f txt tab -s $s_file -al -c both
    python extract.py parallel -sl ${lang^^} -tl EN -i europarl-extract-0.9/txt/ -o corpora/ -f txt tab -s $s_file -al -c both
done
```

Finally, make sure the folder names in ``corpora/parallel/`` are lowercased:

```
cd corpora/parallel/
for f in */; do mv -v "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done
cd ../../../
```

### Our preprocessing

So far the preprocessing of the original repository. Now let's do our own preprocessing.

If you skipped the steps above (recommended), please download the files needed like this:

```
cd europarl-extract
wget https://www.let.rug.nl/rikvannoord/TD/europarl_preprocessed.zip
unzip europarl_preprocessed.zip
cd ../
```

There are a lot of preprocessing steps. They are annotated in the [preprocessing README](preprocessing.md).

You can either follow those instructions step by step, or just simply run:

```
./src/data_preprocessing.sh
```

## Translation 

We want to train a system that can distinguish between original texts (OT), human translations (HT) and machine translations (MT). So far, we only have OT and HT in Europarl, but no MT. We can simply create our own translations by using publicly available multi-lingual MT systems. We use [OPUS-mul-en](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en), [OPUS-en-mul](https://huggingface.co/Helsinki-NLP/opus-mt-en-mul), [NLLB-200-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) and [m2m100-418M](https://huggingface.co/facebook/m2m100_418M). There is a script that lets you easily run these models, if you specify the source and target language. For example, for English to Dutch for the NLLB model:

```
export file="exp_data/europarl/split/train/down/100/nl-en.sent.tab"
python src/translate.py -m nllb -s $file -sl nl -tl en -o ${file}.nllb -tf
```

Note that OPUS, NLLB and m2m all use different language codes, but the Python script normalizes them. If you specify "opus", the script will figure out if you mean opus-en-mul or opus-mul-en by checking if English is the source or the target.

Now say we want translations for the three systems, for all Europarl sentence-level files that we downsampled to 3k lines. This will take hours and needs to run on GPU. You can use a script to make it easier:

```
./src/translate.sh exp_data/europarl/split/train/down/3000/
```

This will only work if the filenames start with the source and target language ISO codes, separated by a dash (e.g. en-nl.sent).

Similarly, we want to get translations for our Europarl dev and test sets, and our MaCoCu and WMT sets:

```
./src/translate.sh exp_data/europarl/split/dev/ euro
./src/translate.sh exp_data/europarl/split/test/ euro
./src/translate.sh exp_data/wmt/ wmt

for type in dev test; do
	for lang in sl hr; do
		./src/translate.sh exp_data/macocu/${lang}/${type}/ macocu
	done
done
```

The MaCoCu train set is left out on purpose for now: translating this will take a lot of time. First figure out if you actually want to do this for all MT systems, and for what size of the files (maybe downsample first).

## Training sets

These files are not training sets yet, as they're all individual sets in one direction, with machine translations. Let's create training sets for the folders we have translations for. We want to create a multi-lingual data set, with all Europarl languages.

Each text can be either OT, HT or MT. However, if we have a parallel sentence pair, we assume that 1 of the 2 sentences is an original text (OT). Therefore, each sentence-pair can have four labels:

* First original, second human translation (first-orig-second-ht)
* First original, second machine translation (first-orig-second-mt)
* Second original, first human translation (second-orig-first-ht)
* Second original, first machine translation (second-orig-first-mt)

We randomize the first/second order so it's not a signal to the model.

Now, we have to decide how and how many translations we add when training the system. We can train on translation from a single system, but also on a mixed data set of translation. We can keep the label division balanced, or use an imbalanced set. We recognize the following options:

* **all-mt**: Add all MTs for each instance: 1-orig-ht, 1-orig-opus, 1-orig-nllb, 1-orig-m2m, 2-orig-ht, 2-orig-opus, ..., etc
* **all-mt-balanced** add all MTs, same as before, but balance the full data set by duplication the orig-ht pairs (e.g. add the original 3 times if we have 3 MT systems). 1-orig-ht, 1-orig-opus, 1-orig-ht, 1-orig-nllb, 1-orig-ht, 1-orig-m2m.
* **single-mt**: Add a single MT system for each instance. 1-orig-ht, 1-orig-opus, 2-orig-ht, 2-orig-opus, etc (useful for cross-mt-system evaluation)
* **random-mt**: Randomize MT selection for each instance: 1-orig-ht, 1-orig-opus, 2-orig-ht, 2-orig-nllb, 3-orig-ht, 3-orig-nllb, etc
* **either-random**: For each original sentence, either add the MT (randomly picked) or HT (and not both), 1-orig-opus, 2-orig-ht, 3-orig-ht, 4-orig-nllb, etc
* **either-single**: Same as above, but always use the same MT system instead, 1-orig-opus, 2-orig-ht, 3-orig-ht, 4-orig-opus, 5-orig-opus, etc

We provide a script that can do this. It reads from a tab-separated file, taking the first two columns. All meta data now gets deleted. It reads all files in a folder. You have to specify the file-type (euro, wmt, macocu) to make sure it reads the correct files. Say we want to create a data set for the setting **random-mt**, given our Europarl folder with translations:

```
export fol="exp_data/europarl/split/train/down/3000/"
python src/create_translation_dataset.py -f $fol -d random-mt -o ${fol}/random-mt.all.clf -ft euro -ti opus nllb m2m
```

We now have a shuffled data set with 3 columns: sentence1, sentence2 and the label (4 options, see above). To make absolutely sure there is no mixup in labels, we run the data through a randomizer that makes sure the first instances have the 4 labels in the same order:

```
python src/order_label.py ${fol}/random-mt.all.clf
```

This gives us the final data set we can use for training in ``${fol}/random-mt.all.clf.ord``.

We want to run this data set creation process for all our Europarl, MaCoCu and WMT data sets. For flexibility, we create data sets with all possible options (random-mt, all-mt, etc). We do this for the train, dev and test sets we have translations for. You can use a helper script to do this for a given folder and identifier (euro, macocu, wmt), given that there exist files in the folder in the correct format for the identifier (e.g. for macocu files that end with .en-${lang} or .${lang}-en). It will read all such files in the folder and combine them all to a data set. However, all the translation files need to exist, otherwise the script will throw an error.

```
./src/create_data_sets.sh exp_data/europarl/split/dev/ europarl
./src/create_data_sets.sh exp_data/europarl/split/test/ europarl
./src/create_data_sets.sh exp_data/macocu/hr/dev/ macocu
./src/create_data_sets.sh exp_data/macocu/hr/test/ macocu
./src/create_data_sets.sh exp_data/macocu/sl/dev/ macocu
./src/create_data_sets.sh exp_data/macocu/sl/test/ macocu
./src/create_data_sets.sh exp_data/wmt/wmt16/ wmt
./src/create_data_sets.sh exp_data/wmt/wmt17/ wmt
./src/create_data_sets.sh exp_data/wmt/wmt18/ wmt
./src/create_data_sets.sh exp_data/wmt/wmt21/ wmt
```

## Training

We want to train a sentence-level XLM-R (base) model for 1 epoch on the **random-mt** Europarl data set we just created. We evaluate on the Europarl full dev set, the MaCoCu Slovene dev set and the Turkish WMT16 set.

When running experiments, we need to specify our settings in a **configuration file**.

The ``src/train.sh`` script always reads from ``config/default.sh``. This contains all the default settings. Please check these default settings carefully!

We also have to specify our own configuration file with the specific settings for this experiment. This is required for each experiment. It will override the default settings.

For this example experiment, I've added ``config/example.sh``. But you can easily use your own file for this.

Run the training (on GPU) as follows:

```
mkdir -p exps
./src/train.sh config/example.sh
```

All experimental files are saved in ``exps/example/``, in different subfolders. In the eval folder you find the experimental results for the 3 to be tested sets.

## Parsing

If you just want to parse a file with an already trained model, use ``src/parse.sh``. You have to specify the model folder, the file with sentence-pairs and the LM identifier.

If you have sentence file SENT, you get the predictions in SENT.pred and the probabilities in SENT.pred.prob.

For example, for the model we just trained:

```
./src/parse.sh exps/example/models/ SENT xlm-roberta-base
```
