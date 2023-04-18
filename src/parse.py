#!/usr/bin/env python

'''Make predictions using a trained finetuned LM on a file of sentences

   Example usage:
   python src/parse.py --model model/ --sent_file sentences.txt'''

import sys
import os
import time
import random as python_random
import argparse
import ast
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, Trainer
from scipy.special import softmax
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from train import process_data
from create_translation_dataset import write_to_file
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Location of the trained model (folder)")
    parser.add_argument("-s", "--sent_file", required=True, type=str,
                        help="Predict on these sentences, not tokenized/processed yet")
    parser.add_argument("-b", "--batch_size", default=64, type=int,
                        help="Batch size during parsing")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file, if not specified add .pred and .pred.prob to sent file")
    parser.add_argument("-lm", "--lm_ident", type=str, default="xlm-roberta-base",
                        help="Language model identifier")
    parser.add_argument("-cm", "--conf_matrix", type=str, default="",
                        help="If added, it's the location where we save the confusion matrix")
    parser.add_argument("-ml", "--max_length", default=128,
                        help="Max length of the inputs")
    parser.add_argument("-pa", "--padding", default="max_length", type=str,
                        help="How to do the padding: max_length (default) or longest")
    parser.add_argument("-se", "--seed", default=2345, type=int,
                        help="Should not matter for prediction, but you can specify it either way")
    # HuggingFace does not allow to only keep a single model throughout training, so you can
    # always resume training. But this is annoying if you really never plan to do that anyway
    # We want to automatically determine the model we use for parsing (if there are two)
    # For that to work, we read the log file and the corresponding metrics. If the best metric is
    # not the last model, we use the first one. We have to read this from a log file, as
    # HuggingFace apparantely does not easily return metrics per epoch in an object
    parser.add_argument("-tl", "--train_log", type=str,
                        help="Location of train log file if we have to find the model")
    args = parser.parse_args()

    # The seed shouldn't matter for only parsing (and doesn't in our experiments)
    # But we set it anyway here so you can experiment with it if you want
    np.random.seed(args.seed)
    python_random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


def get_best_epoch_idx(train_log):
    '''Return 1 if highest epoch was the best (F1 based), else 0'''
    best_f, best_idx, total = 0, 0, 0
    # Loop over the log file and read all the eval_loss lines as a dict
    for line in open(train_log, 'r', encoding="utf-8"):
        if line.strip()[1:].startswith("'eval_loss':"):
            dic = ast.literal_eval(line.strip())
            if dic["eval_f1"] > best_f:
                best_f = dic["eval_accuracy"]
                best_idx = total
            total += 1
    # Best idx was the last one
    if best_idx == total -1:
        return 1
    # Else it was the previous one, so the only one we have left
    return 0


def select_model(mod, train_log):
    '''Select highest or lowest checkpoint based on the logs'''
    subfolders = [f.name for f in os.scandir(mod) if f.is_dir() and f.name.startswith('checkpoint')]
    # No checkpoints, just work with this model
    if not subfolders:
        return mod
    if len(subfolders) == 1:
        # Just return the one model we found
        return mod + "/" + subfolders[0] + "/"
    if len(subfolders) > 2:
        raise ValueError("If you do not specify an actual single model, we can only work with a folder of two checkpoints")
    # Sort checkpoints from low to high
    fol_nums = [[fol, int(fol.split("-")[-1])] for fol in subfolders]
    sort = sorted(fol_nums, key=lambda x: x[1], reverse=False)
    # Read the log file and decide if highest epoch was best
    if not train_log:
        raise ValueError("You need to specify --train_log if you want to automatically find the best model in args.model")
    idx = get_best_epoch_idx(train_log)
    return mod + "/" + sort[idx][0] + "/"


def plot_confusion_matrix(cm, target_names, cm_file):
    '''Given a sklearn confusion matrix (cm), make a nice plo'''
    title='Confusion matrix'
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.savefig(cm_file, format="pdf", bbox_inches="tight")


def evaluate(trainer, in_data, output_file, Y_data, uniq_labels, do_softmax, batch_size, cm_file):
    '''Evaluate a trained model on a dev/test set, print predictions to file possibly'''
    # Actually get the output - also time it
    start = time.time()
    # Make a data loader out of it so we can specify batch size
    test_loader = DataLoader(in_data, batch_size=batch_size, shuffle=False)
    output = trainer.prediction_loop(test_loader, description="prediction")
    end = time.time()
    print (f"Predicting took {end-start} seconds")
    # Sometimes the output is a tuple, take first argument then
    if isinstance(output.predictions, tuple):
        out = output.predictions[0]
    else:
        out = output.predictions

    preds = np.argmax(out, axis=1)
    header = [", ".join(uniq_labels)]
    # If a file was specified, print predictions to this file
    # First convert numbers back to labels
    if output_file:
        out_labels = [uniq_labels[pred] for pred in preds]
        write_to_file(out_labels, output_file)
        # Write probabilities, maybe do softmax first
        if do_softmax:
            out_lines = header + [" ".join([str(x) for x in softmax(row)]) for row in out]
        else:
            out_lines = header + [" ".join([str(x) for x in row]) for row in out]
        write_to_file(out_lines, output_file + '.prob')

    # Print classification report if we have labels
    if Y_data:
        # For a nicer report, convert labels back to strings first
        Y_lab = [uniq_labels[idx] for idx in Y_data]
        pred_lab = [uniq_labels[idx] for idx in preds]
        print ("Classification report:\n")
        print (classification_report(Y_lab, pred_lab, digits=3))
        # Only plot/save confusion matrix if we asked for it
        if cm_file:
            plot_confusion_matrix(confusion_matrix(Y_lab, pred_lab, labels=uniq_labels), uniq_labels, cm_file)


def main():
    '''Main function to parse a new file with a finetuned LM given cmd line arguments'''
    args = create_arg_parser()

    # Set order of labels (important!), as this was automatically determined during training,
    # so we have to use the same order for the predictions to make sense
    # If you run src/order_label.py over your data set, this is always the label order to use
    labels = ["first-orig-second-ht", "first-orig-second-mt",
              "second-orig-first-ht", "second-orig-first-mt"]
    print ("Working with labels:", labels)

    # Read in data. Lots of arguments can be default/empty/false, they only matter
    # for training a model and since we use the same function we have to specify them here
    test_data, Y_test, _ = process_data(args.sent_file, 0, args.lm_ident, args.padding, args.max_length, labels)

    # Select model, see explanation in argparser as to why this complicated procedure is necessary
    mod = select_model(args.model, args.train_log)
    print (f"Do prediction with {mod}")
    # Set up the model and the trainer
    model = AutoModelForSequenceClassification.from_pretrained(mod)
    trainer = Trainer(model=model)

    # If we didn't specify the output file, add .pred and .pred.prob to the sentence file
    if not args.output_file:
        out_file = args.sent_file + '.pred'
    else:
        out_file = args.output_file

    # Run evaluation, this write the predictions to an output file (Y_test is empty)
    # Note that in outfile.prob you get the softmax predictions!
    # We do softmax here as a default, but you could also get the logits (by True -> False)
    # If your sent file contains labels (tab separated), we automatically evaluate the predictions
    evaluate(trainer, test_data, out_file, Y_test, labels, True, args.batch_size, args.conf_matrix)


if __name__ == '__main__':
    # For logging purposes
    print("Generated by command:\npython", " ".join(sys.argv))
    main()
