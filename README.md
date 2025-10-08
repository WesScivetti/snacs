# Multilingual SNACS Classification Experiments

This codebase houses the SNACS classifiers for Chinese, English, Gujarati, Hindi, and Japanese. These results are described in [Scivetti et al. 2025](https://aclanthology.org/2025.coling-main.247/) (COLING 2025).

## Help

If you'd like to use this repo and are having issues, please reach out directly to me via email at wss37@georgetown.edu.

## Results

See [paper](https://aclanthology.org/2025.coling-main.247/) for scores.

## How to Use

You can use this repo to do multiple things, primarily:
 - Train new models on existing SNACS annotated data
 - Run existing models on new annotated or unannotated data

We will start by going over how to run an existing model if you would like to generate predictions on a new file.

## Data Preprocessing

If you would like to run an existing model, you need to have data in the conllulex file format. This is an extension of the CoNLL-U format that is used for SNACS annotations. The actual CoNLL-U-Lex format contains
rich information (see [this link](https://github.com/nert-nlp/streusle/blob/v4.7.1/CONLLULEX.md) for details), but you will just need a shell with the appropriate columns for the script to run correctly. 
To generate such a file, please run preprocess.py in the following way:

```bash
python preprocess.py --input_file 'your_input_file' --input_format 'conllu' --lang en
```

If you are starting from plain text, change the input format to 'plain'. The script will then use [stanza](https://stanfordnlp.github.io/stanza/) to generate a CoNLL-U file before the CoNLL-ULex. 
Please add the two character language code. 


## Running an Existing Model

The first step to running an existing model is preprocessing the data into the conllulex format. That is described above. 

If you'd like to run an existing model, you need to run train.py with the --eval_only or --predict_only flags. Add the file you want to predict on after the --test_file flag. The existing models are in this repo as zip files, except for the english model which is unzipped. 

Alternatively, you can download the models off the huggingface hub:

- [Chinese](https://huggingface.co/WesScivetti/SNACS_Chinese)
- [English](https://huggingface.co/WesScivetti/SNACS_English)
- [Gujarati](https://huggingface.co/WesScivetti/SNACS_Gujarati)
- [Hindi](https://huggingface.co/WesScivetti/SNACS_Hindi)
- [Japanese](https://huggingface.co/WesScivetti/SNACS_Japanese)

You will want to load those models in the same way as any other huggingface model. After loading the model, you can run the following script:

```bash
python train.py --predict_only --test_file 'your_test_file.conllulex' --lang en
```

If this works, it should write the results to '{your_test_file}_predicted.conllulex' with the predicted lextags in the last column. 


## Training a Model

If you'd like to train a model, then simply run train.py and supply a train, dev and test file, as well as a pretrained model name. Most hyperparameters have defaults, but you can set those optionally with the available flags. All of the flags are listed below:
 - --model_name: specify the pretrained model name that you want to load off huggingface.
 - --loss_fn: if you want to add a custom loss function here. Probably won't need to do this.
 - --file: The training file that you want to train on. Should be located in the data/splits/ subdirectory.
 - --learning_rate
 - --batch_size
 - --epochs
 - --weight_decay
 - --freeze: if you want to freeze the parameters of the pretrained model (will train faster)
 - --warmup_steps
 - --lr_scheduler: scheduler type, the two we use are "linear" and "cosine".
 - --test_file: The file to test or predict on. 
 - --dev_file: The file which hyperparameters are tuned on, used for evaluation during training
 - --extra_file: Additional training data file, probably from another language.
 - --extra_dev_file: Only used if you'd like to evaluate on multiple languages simultaneously.
 - --extra_test_file: Only used if you'd like to evaluate on multiple languages simultaneously.
 - --lang: The two character language code.
 - --multilingual: If supplying an extra lang file, put true to include that language in eval. Otherwise it will only test on original lang
 - --do_sweep: run Bayesian hyperparameter sweep. Need to login to wandb prior to this for this to work.
 - --eval_only: Include if you want to load an existing model and evaluate on the provided test set. No training involved.
 - --predict_only: Include if you want to load an existing model and predict on the provided test set. Evaluation is not provided (appropriate for data where no gold labels exist), No training involved.
 - --use_best_hypers: Use the best performing hyperparameters from our sweeps. Overwrites flags from above.
 - --extra_lang: Two letter language code of the additional language which is being added. 
 
If you'd like to use the hyperparameters that we used in the paper, use the --use_best_hypers flag which will override the necessary hyperparameters. 

If you'd like to supply additional supplemental training data, you can specify that with the --extra_file flag.


# Data Description

The following files are located in the data directory:
- de-lp-new.conllulex (German SNACS in conllulex. See [the paper](https://link.springer.com/article/10.1007/s13218-021-00712-y) for description. NOT up to date with SNACS 2.6)
- de-lp.conllulex (German SNACS in conllulex. See [the paper](https://link.springer.com/article/10.1007/s13218-021-00712-y) for description. NOT up to date with SNACS 2.6)
- en-lp.conllu (English LP, base conllu with no SNACS columns. Up to date.)
- **en-lp_c.conllulex** (English LP, all chapters combined and chapter metadata added. Up to date in SNACS 2.6)
- en-pastrie.conllulex (English PASTRIE full corpus. Not used in experiments for paper. See [the PASTRIE repo](https://github.com/nert-nlp/pastrie) and [the paper](https://aclanthology.org/2020.law-1.10/))
- **en-streusle.conllulex** (English STREUSLE corpus. Used as supplemental data in some experiments. For train,dev, and test splits, see [the STREUSLE repo](https://github.com/nert-nlp/streusle/))
- **gu-lp_c.conllulex** (Gujarati LP with chapter metadata added. Up to date. See [the paper](https://aclanthology.org/2023.findings-acl.696/))
- **hi-lp_c.conllulex** (Hindi LP with chapter metadata added. Up to date. See [the paper](https://aclanthology.org/2022.lrec-1.612/))
- **jp-lp_c.conllulex** (Japanese LP excerpt with chapter metadata added. Up to date. See [the paper](https://aclanthology.org/2024.lrec-main.839/))
- **zh-lp_c.conllulex** (Chinese LP with chapter metadata added. Up to date. See [the paper](https://aclanthology.org/2020.lrec-1.733/))

These resources are generally a subset of those listed under the CARMLS datasets in [this repo](https://github.com/carmls/datasets).

LP = _The Little Prince_ (translated from French _Le Petit Prince_)

## Statistics

The 6 datasets listed in bold were used by [Scivetti et al. 2025](https://aclanthology.org/2025.coling-main.247/), which gave the following summary:
<img width="1054" alt="image" src="https://github.com/user-attachments/assets/7d2cd88a-b569-4bba-b3ce-133a1ba05870" />

## File Format

The `.conllulex` format combines 10 columns of the `.conllu` format for Universal Dependencies with additional columns for lexical semantic annotation. It was first used for STREUSLE and is described [here](https://github.com/nert-nlp/streusle/blob/v4.7.1/CONLLULEX.md).

## Chapter Metadata
Most of the LP files above have chapter metadata added in. This is because the load_data script uses this metadata to conduct train, dev, and test splits by chapter. 




