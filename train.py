from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from load_data import tokenize_and_align, get_ss_frequencies, inversify_freqs
import numpy as np
import evaluate
import random
import argparse
import os
from torch.nn import CrossEntropyLoss
import torch
import sys
from collections import defaultdict
import wandb
import json
import glob
import shutil

# random seed
random.seed(42)

# make logs dir
if not os.path.exists("logs"):
    os.makedirs("logs")

# setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seqeval = evaluate.load("seqeval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def load_data(file: str, tokenizer: AutoTokenizer, id_to_label = None, label_to_id = None, freqs=None):
    """Load data from file and tokenize it."""
    res = tokenize_and_align(file, tokenizer)

    # potentially, we could recalculate frequencies with extra languages included too. Not sure if that's a good idea though. For now freqs just on the first target lang
    if not freqs:
        freqs = get_ss_frequencies(res)
        # print(freqs["lt"]["B-p.Cost-Extent"])

    else:
        #need to combine frequencies of files to get the inverse freqs right
        old_freqs = freqs
        new_freqs = get_ss_frequencies(res)

        #make a new freqs to house combination of freqs
        freqs = {"lt": {}, "ss": {}, "ss2": {} }
        for tag_type in ["lt", "ss", "ss2"]:
            all_tags = list(set(list(old_freqs[tag_type].keys()) + list(new_freqs[tag_type].keys())))

            for tag in all_tags:
                comb = old_freqs[tag_type][tag] + new_freqs[tag_type][tag]

                #idk why this would happen but it did >:( now I'm making sure on zero counts get in there
                if comb > 0:
                    freqs[tag_type][tag] = comb

    # if label-id mapping exists from previous language file, can use that
    # make label-id mapping if doesn't exist
    if not id_to_label and not label_to_id:
        label_to_id = defaultdict(lambda: -100)
        label_to_id["None"] = -100
        id_to_label = defaultdict(lambda: 'O')
        id_to_label[-100] = "None"

    # convert labels to ids
    for sent, mask, label, lexlemma in res: #removed split
        for i in range(len(sent)):
            if mask[i]:
                if label[i] not in label_to_id:
                    id = len(label_to_id)
                    label_to_id[label[i]] = id
                    id_to_label[id] = label[i]

    res2 = []
    lang_code = file.split("/")[-1].split("-")[0]

    # convert labels to ids
    
    for sent, mask, label, lexlemma in res: #removed split
        label = [label_to_id[x] for x in label]
        res2.append({
            'input_ids': sent,
            'mask': mask,
            'labels': label,
            'lang': lang_code,
            'lexlemma': lexlemma
        })
        
    print(f"{len(label_to_id)} labels.")
    print(label_to_id)

    # shuffle
    #will probably need to change to a train/test split

    random.seed(1000)
    random.shuffle(res2)

    return res2, label_to_id, id_to_label, freqs

def combine_datasets(file_list: list, train_only=False):
    """basically, reads multiple language files in and then combines them into one larger dataset. Useful if you want to """
    return


def compute_metrics(p, id_to_label, eval_dataset):
    """Compute metrics for evaluation."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # make human readable
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    ## NEEDDS FIXING FOR TEST FILES
    # log predictions to file (same id as wandb run)
    # with open(f"logs/{wandb.run.id}.json", "a") as f:
    #
    #     # log id_to_label mapping
    #     if os.path.getsize(f"logs/{wandb.run.id}.json") > 0:
    #         f.write("\n")
    #
    #     # collect predictions
    #
    #     sents = []
    #     for i in range(len(predictions)):
    #         sents.append({
    #             'input_ids': eval_dataset[i]['input_ids'],
    #             'prediction': true_predictions[i],
    #             'label': true_labels[i],
    #             'lexlemma': eval_dataset[i]['lexlemma'],
    #         })
    #
    #     # dump to file
    #     json.dump(sents, f)

    # print("LABELS", true_labels, file=sys.stderr)
    true_scenes = []
    true_functions = []
    for batch in true_labels:
        batch_s = []
        batch_f = []
        for lab in batch:
            if "-" not in lab:
                batch_s.append(lab)
                batch_f.append(lab)
            else:
                bio = lab.split("-")[0]
                scene = lab.split("-")[1]
                func = lab.split("-")[2]
                true_scene = bio + "-" + scene
                true_func = bio + "-" + func
                batch_s.append(true_scene)
                batch_f.append(true_func)
        true_scenes.append(batch_s)
        true_functions.append(batch_f)

    pred_scenes = []
    pred_functions = []

    for batch in true_predictions:
        batch_s = []
        batch_f = []
        for lab in batch:
            if "-" not in lab:
                batch_s.append(lab)
                batch_f.append(lab)
            else:
                bio = lab.split("-")[0]
                scene = lab.split("-")[1]
                func = lab.split("-")[2]
                pred_scene = bio + "-" + scene
                pred_func = bio + "-" + func
                batch_s.append(pred_scene)
                batch_f.append(pred_func)
                
        pred_scenes.append(batch_s)
        pred_functions.append(batch_f)



    # print("TRUE SCENES", true_scenes, file=sys.stderr)
    # print("TRUE FUNCTIONS", true_functions, file=sys.stderr)

    # compute metrics
    results = seqeval.compute(predictions=true_predictions, references=true_labels, scheme="IOB2")

    #compute scene results
    scene_results = seqeval.compute(predictions=pred_scenes, references=true_scenes, scheme="IOB2")

    #compute function results
    func_results = seqeval.compute(predictions=pred_functions, references=true_functions, scheme="IOB2")

    ret = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "scene_precision": scene_results["overall_precision"],
        "scene_recall": scene_results["overall_recall"],
        "scene_f1": scene_results["overall_f1"],
        "scene_accuracy": scene_results["overall_accuracy"],
        "funct_precision": func_results["overall_precision"],
        "funct_recall": func_results["overall_recall"],
        "funct_f1": func_results["overall_f1"],
        "funct_accuracy": func_results["overall_accuracy"],
    }

    #log the metrics
    # wandb.log({
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # })

    # add metrics for each label
    for key in results:
        if isinstance(results[key], dict) and results[key]["number"] != 0:
            ret[key] = results[key]
            
    ##  NEEDS FIXING FOR TEST DATASET
    # acc for each lexlemma type
    # lexlemma = defaultdict(lambda: {"correct": 0, "total": 0})
    # for i in range(len(predictions)):
    #     for j in range(len(predictions[i])):
    #         if labels[i][j] == -100 or labels[i][j] == 1:
    #             continue
    #         lexlemma[eval_dataset[i]['lexlemma'][j]]["total"] += 1
    #         if predictions[i][j] == labels[i][j]:
    #             lexlemma[eval_dataset[i]['lexlemma'][j]]["correct"] += 1
    #
    # # calculate acc and put in ret
    # for key in lexlemma:
    #     ret[f"{key}.acc"] = lexlemma[key]["correct"] / lexlemma[key]["total"]

    return ret

# custom trainer which is used for custom weighted loss function
# class MyTrainer(Trainer):
#     def add_freqs(self, freqs):
#         self.freqs = freqs
#         self.inv_freqs = inversify_freqs(self.freqs)
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         custom loss function which overwrites the standard compute_loss function. We use this to implement the weighted CE loss
#         """
#
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         logits = logits.view(
#             -1, logits.shape[-1]
#         )  # have to reshape to (batch_size * sequence_length, # labels)
#
#         num_labels = logits.size(1)
#
#         #TO DO: compute weights based on frequency of relative labels in input
#         #below is just some random experiments with changing the weights to see if there was significant effect
#
#         weights = [1] * num_labels
#
#         weights2 = [.0001] +list(self.inv_freqs["lt"].values())
#         weights[1] = 0.1 #downweighting label "O" which seems to be label 1 almost always
#         weights[0] = 0.0001 #downweighting label "-100" ... not sure if would ever matter
#
#         # print(len(weights), len(weights2), file=sys.stderr)
#         assert len(weights2) == len(weights)
#
#         weights = [float(w) for w  in weights]
#
#         weights2 = [float(w) for w in weights2]
#
#         weights = torch.tensor(weights).to(DEVICE)
#         weights2 = torch.tensor(weights2).to(DEVICE)
#
#
#         labels = labels.view(-1) #batch_size * sequence length
#
#         loss_fn = CrossEntropyLoss(weight=weights2)
#         loss = loss_fn(logits, labels)
#
#         if return_outputs:
#             return loss, outputs
#         else:
#             return loss


# model training
def train(
    model_name: str, #need - added
    file: str, #need - added
    learning_rate: float, #don't need
    batch_size: int, #don't need
    epochs: int, #don't need
    weight_decay: float, #don't need
    freeze: bool, #need - added
    test_file: str, #need
    extra_file: str, #need
    multilingual: bool, #need
    loss_fn: str #need - added
):
    """Train model."""

    # update summary for wandb
    command_line_args = locals()
    # print(locals())

    # load data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data, label_to_id, id_to_label, freqs = load_data(f"data/splits/{file}", tokenizer)

    # could alter this to take a list of extra files so that it could be as many as you want.
    if extra_file:
        #for ex_file in extra_file: do this iteratively, add each extra file onto eachother, take the new label_to_id etc
        extra_data, label_to_id, id_to_label, freqs = load_data(f"data/{extra_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #use the existing id_to_label and just add to them


    if test_file:
        test_data, _, _, _ = load_data(f"data/{test_file}", tokenizer) #don't need label to id for this


    # load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    print("NUM labels", len(label_to_id), file=sys.stderr)

    # freeze layers
    if freeze:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # set up trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    # split the file into train and eval if not separate eval file
    if not test_file:

        #this asks if you want to train and test on combination of languages or just train on combination and test on single
        #for example: you could train on en + hi and test on en + hi (multilingual = True)
        #or you could train on en + hi and test on hi only (multilingual = False)
        if extra_file:
            if multilingual:
                data = data + extra_data #combine first then split
                train_dataset = data[len(data) // 5:]
                eval_dataset = data[:len(data) // 5]
            else:
                train_dataset = data[len(data) // 5:] + extra_data #combine extra only with training
                eval_dataset = data[:len(data) // 5]

        #this is most simple case: 1 file, split it into train + eval
        else:
            train_dataset = data["train"]
            eval_dataset = data["dev"]
            test_dataset = data["test"]

    #if you supply a test file separately, you will test on that, and train on training data
    else:
        #if you supply extra data, add that into training too
        if extra_file:
            data = data + extra_data

        train_dataset = data
        eval_dataset = test_data

    # set up trainer
    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": lambda x: compute_metrics(x, id_to_label, eval_dataset),
    }
    trainer = None
    if loss_fn == "weighted":
        trainer = MyTrainer(**trainer_args, save_strategy="no", save_model=False)
        trainer.add_freqs(freqs)
    else:
        #adding in manually supplied stuff
        trainer = Trainer(**trainer_args)

    # update
    run = wandb.init(project="huggingface")
    run.summary.update(command_line_args)

    # train
    trainer.train()


def train2(config=None):
    """Train model -- set up for wandb hyperparam sweep"""



    wandb.init()
    config = wandb.config

    filename = config.file

    lang = filename.split("-")[0]

    print("LANGUAGE", lang, file=sys.stderr)

    try:
        with open("/best_model/" + lang + "/metric/f1.txt", "r") as f:
            best_metric = float(f.read().strip())
    except FileNotFoundError:
        best_metric = None

    # load data
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data, label_to_id, id_to_label, freqs = load_data(f"data/splits/{config.file}", tokenizer)

    # could alter this to take a list of extra files so that it could be as many as you want.
    if config.extra_file:
        #for ex_file in extra_file: do this iteratively, add each extra file onto eachother, take the new label_to_id etc
        extra_data, label_to_id, id_to_label, freqs = load_data(f"data/{config.extra_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #use the existing id_to_label and just add to them

    if config.dev_file:
        dev_data, _, _, _ = load_data(f"data/splits/{config.dev_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this


    if config.test_file:
        test_data, _, _, _ = load_data(f"data/splits/{config.test_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this


    # load model
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    print("NUM labels", len(label_to_id), file=sys.stderr)

    # freeze layers
    if config.freeze:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # set up trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.scheduler_type,  # dynamic scheduler type
        warmup_steps=config.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True, #change to True
        push_to_hub=False
    )

    # split the file into train and eval if not separate eval file
    if not config.test_file:
        #this asks if you want to train and test on combination of languages or just train on combination and test on single
        #for example: you could train on en + hi and test on en + hi (multilingual = True)
        #or you could train on en + hi and test on hi only (multilingual = False)
        if config.extra_file:
            if config.multilingual:
                data = data + extra_data #combine first then split
                train_dataset = data[len(data) // 5:]
                eval_dataset = data[:len(data) // 5]
            else:
                train_dataset = data[len(data) // 5:] + extra_data #combine extra only with training
                eval_dataset = data[:len(data) // 5]

        #this is most simple case: 1 file, split it into train + eval
        else:
            train_dataset = data[len(data) // 5:]
            eval_dataset = data[:len(data) // 5]

    #if you supply a test file separately, you will test on that, and train on training data
    else:
        #if you supply extra data, add that into training too
        if config.extra_file:
            data = data + extra_data

        train_dataset = data
        eval_dataset = dev_data
        test_dataset = test_data



    # set up trainer
    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": lambda x: compute_metrics(x, id_to_label, eval_dataset),
    }
    trainer = None
    if config.loss_fn == "weighted":
        trainer = MyTrainer(**trainer_args, save_strategy="no", save_model=False)
        trainer.add_freqs(freqs)
    else:
        #adding in manually supplied stuff
        trainer = Trainer(**trainer_args)

    # train
    trainer.train()

    wandb.log({"evaluation_loss": trainer.evaluate()['eval_loss']})

    best_f1 = trainer.evaluate()['eval_f1']

    test_results = trainer.predict(test_dataset)

    wandb.log(test_results.metrics)

    if best_metric is None or best_f1 > best_metric:
        with open("./best_model/" + lang + "/metric/f1.txt", "w") as f:
            f.write(str(best_f1))

        # shutil.rmtree("./best_model/" + lang + "/model/")

        for f in glob.glob("./best_model/" + lang + "/model/.*"):
            os.remove(f)
        
        model_path = "./best_model/" + lang + "/model/"
        trainer.save_model(model_path)





    wandb.finish()

    #remove model checkpoints, already saved the best one
    shutil.rmtree("./models/")




def hyper_sweep(args):

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval/f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-6,
                'max': 1e-3
            },
            'batch_size': {
                'values': [4, 8, 16, 24]
            },
            'weight_decay': {
                'values': [0.0, 0.01, 0.1]
            },
            'file': {
                'value': args.file
            },
            'model_name': {
                'value': args.model_name
            },
            'freeze': {
                "value": False
            },
            'loss_fn': {
                "value": None
            },
            'test_file': {
                "value": args.test_file
            },
            'extra_file': {
                "value": args.extra_file
            },
            'dev_file': {
                "value": args.dev_file
            },
            'multilingual': {
                "value": False
            },
            'epochs': {
                "value": 10
            },
            'scheduler_type': {
                'values': ['linear', 'cosine']
            },
            'warmup_steps': {
                'min': 0,
                'max': 500
            },
        }
    }


    sweep_id = wandb.sweep(sweep_config, project="huggingface")

    wandb.agent(sweep_id, train2, count=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--loss_fn", type=str, default=None) #overwritten by wandb sweep
    parser.add_argument("--file", type=str, default="en-test.conllulex")
    parser.add_argument("--learning_rate", type=float, default=2e-5) #overwritten by wandb sweep
    parser.add_argument("--batch_size", type=int, default=16) #overwritten by wandb sweep
    parser.add_argument("--epochs", type=int, default=10) #overwritten by wandb sweep
    parser.add_argument("--weight_decay", type=float, default=0.01) #overwritten by wandb sweep
    parser.add_argument("--freeze", action="store_true") #overwritten by wandb sweep
    parser.add_argument("--test_file", type=str, default=None, help="Need to put file for test split")
    parser.add_argument("--dev_file", type=str, default=None, help="Need to put file for dev split")
    parser.add_argument("--extra_file", type=str, default=None, help="If you want to add an extra file to add more data during the fine-tuning stage. Evaluation is still only perfomed on the original file test split.")
    parser.add_argument("--multilingual", action="store_true", help="If supplying an extra lang file, put true to include that language in eval. Otherwise it will only test on original lang")
    
    args = parser.parse_args()

    # train(**vars(args))
    hyper_sweep(args)


if __name__ == "__main__":
    main()
