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

## Training

Say we want to train a sentence-level XLM-R (base) model for 3 epochs on the combined europarl data, but downsampled to 1,000 instances per language. We evaluate on the Europarl Bulgarian dev set, the MaCoCu dev sets for Slovene and Croatian, and the Turkish WMT16 set.

When running experiments, we need to specify our settings in a **configuration file**.

The ``src/train.sh`` script always reads from ``config/default.sh``. This contains all the default settings. Please check these default settings carefully!

We also have to specify our own configuration file with the specific settings for this experiment. This is required for each experiment. It will override the default settings.

For this example experiment, I've added ``config/example.sh``. But you can easily use your own file for this.

Run the training (on GPU) as follows:

```
mkdir -p exps
./src/train.sh config/example.sh
```

All experimental files are saved in ``exps/example/``, in different subfolders. In the eval folder you find the experimental results for the 4 to be tested sets, which all should have between 60-70% accuracy.

## Parsing

If you just want to parse a file with an already trained model, use ``src/parse.sh``. You have to specify the model folder, the file with sentence-pairs and the LM identifier.

If you have sentence file SENT, you get the predictions in SENT.pred and the probabilities in SENT.pred.prob.

For example, for the model we just trained:

```
./src/parse.sh exps/example/models/ SENT xlm-roberta-base
```
