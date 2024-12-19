# Multilingual SNACS Classification Experiments

This codebase houses the SNACS classifiers for Chinese, English, Gujarati, Hindi, and Japanese. These results are described in [Scivetti et al. 2025](https://people.cs.georgetown.edu/nschneid/p/xlingsnacs.pdf) (COLING 2025).

## Results

See model finetuning runs on [Weights & Biases](https://wandb.ai/nert/huggingface?nw=nwuserwss37).


**English**
| Model | W&B | Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `roberta-large` + Linear | `polished-plasma-24` | `en-streusle` | 70.9 | 72.3 | 71.6 |
| `roberta-base` + Linear | `feasible-dragon-20` | `en-streusle` | 77.0 | 79.6 | **78.2** |
| Liu et al. (2021) | | `en-streusle` | | | 70.9 |
| Schneider et al. (2018) | | `en-streusle` | | | 55.7 |
| `bert-base-cased` + Linear | `confused-elevator-22` | `en-lp` | 67.4 | 70.1 | 68.7 |
| `roberta-base` + Linear | `pleasant-salad-21` | `en-lp` | 66.8 | 69.4 | 68.1 |
| `roberta-base` + Linear | `youthful-frog-30` | `en-pastrie` | 53.9 | 57.7 | 55.7 |

**Hindi**
| Model | W&B | Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `neuralspace-reverie/indic-transformers-hi-roberta` + Linear | `soft-butterfly-33` | `hi-lp` | 61.3 | 63.3 | 62.3 |

## Replicating Results From the Paper

### Single Model (Optimal Hyperparameters)
We report scores for the best performing models. To fine-tune a model with the hyperparameters we found to be optimal, run the following code:

Each of these fine-tuning runs should take well under an hour on a GPU.

### Hyperparameter Sweep
In order to achieve the results above, we tuned hyperparameters across 50-100 runs in each classification setting. To run a hyperparameter tuning sweep yourself, run the following command:

Example usage:


## File Description






