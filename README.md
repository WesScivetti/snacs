# Multilingual SNACS Classification Experiments

This codebase houses the SNACS classifiers for Chinese, English, Gujarati, Hindi, and Japanese. These results are described in [Scivetti et al. 2025](https://people.cs.georgetown.edu/nschneid/p/xlingsnacs.pdf) (COLING 2025).

## Results

See [paper](https://people.cs.georgetown.edu/nschneid/p/xlingsnacs.pdf) for scores.

## How to Use

You can use this repo to do multiple things, primarily:
 - Train new models on existing snacs annotated data
 - Run existing models on new annotated or unannotated data

We will start by going over how to run an existing model if you would like to generate predictions on a new file.

## Data Preprocessing

If you would like to run an existing model, you need to have data in the conllulex file format. This is an extension of the CoNLL-U format that is used for SNACS annotations. The actual CoNLL-ULex format contains
rich information (see [this link](https://github.com/nert-nlp/streusle/blob/master/CONLLULEX.md) for details), but you will just need a shell with the appropriate columns for the script to run correctly. 
To generate such a file, please run preprocess.py in the following way:

```bash
python preprocess.py --input_file 'your_input_file' --input_format 'conllu' --lang en
```

If you are starting from plain text, change the input format to 'plain'. The script will then use [stanza](https://stanfordnlp.github.io/stanza/) to generate a CoNLL-U file before the CoNLL-ULex. 
Please add the two character language code. 


## Running an Existing Model

The first step to running an existing model is preprocessing the data into the 

If you'd like to run an existing model, you need to run train.py with the --eval_only or --predict_only flags. Add the file you want to predict on after the --test_file flag.

```bash
python train.py --predict_only --test_file 'your_test_file.conllulex' --lang en
```

If this works, it should write the results to '{your_test_file}_predicted.conllulex' with the predicted lextags in the last column. 

## Training a Model

If you'd like to train a model, then simply run train.py and supply a train, dev and test file, as well as a pretrained model name. Most hyperparameters have defaults, but you can set those optionally with the available flags. If you'd like to use the hyperparameters that we used in the paper, use the --use_best_hypers flag which will override the necessary hyperparameters. 

If you'd like to supply additional supplemental training data, you can specify that with the --extra_file flag.




