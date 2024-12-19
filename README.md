# Multilingual SNACS Classification Experiments

This codebase houses the SNACS classifiers for Chinese, English, Gujarati, Hindi, and Japanese. These results are described in [Scivetti et al. 2025](https://people.cs.georgetown.edu/nschneid/p/xlingsnacs.pdf) (COLING 2025).

## Results

See [paper](https://people.cs.georgetown.edu/nschneid/p/xlingsnacs.pdf) for scores.



## Replicating Results From the Paper

### Single Model (Optimal Hyperparameters)
We report scores for the best performing models. To fine-tune a model with the hyperparameters we found to be optimal, run the following code:

```
python train.py --model_name [MODEL NAME] --file [GOLD TRAIN FILE] --dev_file [DEV FILE] --test_file [TEST FILE] --extra_dev_file [EXTRA DEV FILE] --extra_test_file [EXTRA TEST FILE] --extra_file [EXTRA TRAIN FILE] --do_sweep
```


Each of these fine-tuning runs should take well under an hour on a GPU. There are a bunch of optional arguments:


### Hyperparameter Sweep
In order to achieve the results above, we tuned hyperparameters across 50-100 runs in each classification setting. To run a hyperparameter tuning sweep yourself, run the following command:
```
python train.py --model_name [MODEL NAME] --file [GOLD TRAIN FILE] --dev_file [DEV FILE] --test_file [TEST FILE] --extra_dev_file [EXTRA DEV FILE] --extra_test_file [EXTRA TEST FILE] --extra_file [EXTRA TRAIN FILE] --do_sweep
```

The main thing to note is that you need to add the --do_sweep flag, otherwise all the arguments are the same as with a single run.

Example usage:


## File Description

descriptive_stats.py: Used for printing some basic descriptive stats for the SNACS datasets.
add_chapter_metadata.py: Added chapter information to the metadata for the conllulex files, coming from sentence IDs. Don't need to run again as the versions with the chapter metadata are in the data folder already.






