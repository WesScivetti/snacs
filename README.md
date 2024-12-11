# snacs

Models for parsing SNACS datasets. See [Schneider et al. (2018)](https://aclanthology.org/P18-1018/).

## Results

See model finetuning runs on [Weights & Biases](https://wandb.ai/nert/huggingface?nw=nwuserwss37).


**English**
| Model | W&B | Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `bert-base-cased` + Linear | `polished-plasma-24` | `en-streusle` | 70.9 | 72.3 | 71.6 |
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
