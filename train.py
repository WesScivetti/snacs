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
# import wandb
import json
import glob
import shutil

# random seed
# random.seed(42)

# make logs dir
if not os.path.exists("logs"):
    os.makedirs("logs")

# setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seqeval = evaluate.load("seqeval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"








def load_data(file: str, tokenizer: AutoTokenizer, id_to_label = None, label_to_id = None, freqs=None, shuffle=True):
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
                # print(new_freqs[tag_type][tag])

                comb = 0

                if tag in old_freqs[tag_type]:
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

    if shuffle:
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


def load_trained_model(
        lang:str,
        train_file:str,
        dev_file:str,
        test_file:str,
        do_eval=False):
    """
    Used for loading a saved model and running it for evaluations or predictions on a new dataset. Best models can be found in the best_models/
    subdirectory of this repo. If you just want to load a model and get some snacs predictions, I recommend using one of these.
    lang: the language that you want to load the best model for
    test_file: the file that will be evaluated, predicted on
    do_eval: if you want to eval against existing labels, or just predict. If using a dataset with no snacs annotations, set to False
    """

    print(train_file)

    lang_path = "./best_models/" + lang + "/"
    model_path = lang_path + "model/"

    training_arg_path = model_path + "training_args.bin"

    training_args = torch.load(training_arg_path)

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # l2id_path = lang_path + 'label2id.json'
    # with open(l2id_path, 'r') as file:
    #     label_to_id = json.load(file)
    #
    # id2l_path = lang_path + 'id2label.json'
    # with open(id2l_path, 'r') as file:
    #     id_to_label = json.load(file)
    #     id_to_label = {int(k): v for k,v in id_to_label.items()}
    #
    # freq_path = lang_path + 'freqs.json'
    # with open(freq_path, 'r') as file:
    #     freqs = json.load(file)


    data, label_to_id, id_to_label, freqs = load_data(f"data/splits/{train_file}", tokenizer)

    print("NUM labels just train", len(label_to_id), file=sys.stderr)

    if dev_file:
        dev_data, _, _, _ = load_data(f"data/splits/{dev_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this

    print("NUM labels after dev", len(label_to_id), file=sys.stderr)

    if test_file:
        test_data, _, _, _ = load_data(f"{test_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this

    print("NUM labels after test", len(label_to_id), file=sys.stderr)


    # l2id_path = lang_path + 'label2id.json'
    # with open(l2id_path, 'r') as file:
    #     label_to_id = json.load(file)
    #
    # id2l_path = lang_path + 'id2label.json'
    # with open(id2l_path, 'r') as file:
    #     id_to_label = json.load(file)
    #     id_to_label = {int(k): v for k,v in id_to_label.items()}
    #
    # freq_path = lang_path + 'freqs.json'
    # with open(freq_path, 'r') as file:
    #     freqs = json.load(file)

    print("NUM labels after test", len(label_to_id), file=sys.stderr)


    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )


    print(test_data[0])

    trainer_args = {
        "model": model,
        "args": training_args,
        "eval_dataset": test_data,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": lambda x: compute_metrics(x, id_to_label, test_data),
    }
    
    trainer = Trainer(**trainer_args)



    if do_eval:
        res = trainer.predict(test_data)
        print(res.metrics)
    else:
        res = trainer.predict(test_data)


    predicted_labels = np.argmax(res.predictions, axis=2)

    # print(test_data[0].shape, file=sys.stderr)
    # print(predicted_labels.shape, file=sys.stderr)
    print(predicted_labels[3], file=sys.stderr)
    print(len(test_data))
    print(test_data[3]["input_ids"], file=sys.stderr)
    print(tokenizer.convert_ids_to_tokens(test_data[3]["input_ids"]))
    print(test_data[3]["labels"], file=sys.stderr)
    print(test_data[3]["mask"], file=sys.stderr)
    # print(predicted_labels[0].shape, file=sys.stderr)
    # print(predicted_labels[0][0], file=sys.stderr)
    # print(predicted_labels[0][0].shape, file=sys.stderr)


def write_back_to_file(test_file, test_data, predictions):
    """
    function for writing the predictions back to a file given the predictions on a dataset
    """

    conllulex = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps',
                 'misc', 'smwe', 'lexcat', 'lexlemma', 'ss', 'ss2', 'wmwe', 'wcat', 'wlemma', 'lextag']

    new_file_name = test_file.split(".") + "_predictions.conllulex"

    assert len(test_data) == len(predictions) #need to have same number of rows

    with open(test_file, 'r') as fin:
        sent_num = 0
        for sent in tqdm(conllu.parse_incr(fin, fields=conllulex)):
            text = sent.metadata['text']
            gold_mask = test_data[sent_num][""]



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
    dev_file: str,
    extra_file: str, #need
    warmup_steps = 100,
    lr_scheduler = "cosine",
    multilingual = False, #need
    loss_fn = None, #need - added
    do_sweep = False,
    number = 0
):
    """Train model."""

    # # update summary for wandb
    # command_line_args = locals()

    # load data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data, label_to_id, id_to_label, freqs = load_data(f"data/splits/{file}", tokenizer)
    print("NUM labels just train", len(label_to_id), file=sys.stderr)

    # could alter this to take a list of extra files so that it could be as many as you want.
    if extra_file:
        #for ex_file in extra_file: do this iteratively, add each extra file onto eachother, take the new label_to_id etc
        extra_data, label_to_id, id_to_label, freqs = load_data(f"data/splits/{extra_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #use the existing id_to_label and just add to them

    if dev_file:
        dev_data, _, _, _ = load_data(f"data/splits/{dev_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this

    print("NUM labels after dev", len(label_to_id), file=sys.stderr)

    if test_file:
        test_data, _, _, _ = load_data(f"data/splits/{test_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this
    print("NUM labels after test", len(label_to_id), file=sys.stderr)

    # load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    print("NUM labels", len(label_to_id), file=sys.stderr)
    print("NUM labels", id_to_label, file=sys.stderr)
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
        lr_scheduler_type=lr_scheduler,  # dynamic scheduler type
        warmup_steps=warmup_steps,
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
    if loss_fn == "weighted":
        trainer = MyTrainer(**trainer_args, save_strategy="no", save_model=False)
        trainer.add_freqs(freqs)
    else:
        #adding in manually supplied stuff
        trainer = Trainer(**trainer_args)

    # # update
    # run = wandb.init(project="huggingface")
    # run.summary.update(command_line_args)

    # train
    trainer.train()

    res = trainer.evaluate(eval_dataset=test_dataset)

    model_path = f"./models/final/{number}/"


    if not os.path.exists(model_path):
        os.makedirs(model_path)


    with open(model_path + "id2label.json", "w") as file:
        json.dump(id_to_label, file)

    with open(model_path + "label2id.json", "w") as file:
        json.dump(label_to_id, file)

    with open(model_path + "freqs.json", "w") as file:
        json.dump(freqs, file)



    trainer.save_model(model_path)
    trainer.save_metrics(f"final/{number}/", res)


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
        extra_data, label_to_id, id_to_label, freqs = load_data(f"{config.extra_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label) #use the existing id_to_label and just add to them



    if config.dev_file:
        dev_data, _, _, _ = load_data(f"data/splits/{config.dev_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this


    if config.test_file:
        test_data, _, _, _ = load_data(f"data/splits/{config.test_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this

    if config.extra_dev_file:
        extra_dev_data, _, _, _ = load_data(f"data/splits/{config.extra_dev_file}", tokenizer, label_to_id=label_to_id,
                                      id_to_label=id_to_label, freqs=freqs)  # don't need label to id for this
    
    if config.extra_test_file:
        extra_test_data, _, _, _ = load_data(f"data/splits/{config.extra_test_file}", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label, freqs=freqs) #don't need label to id for this


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


        else:
            if config.multilingual:
                #this is the setting where we train and test on everything at once!!
                #set all existing variables to none
                data, label_to_id, id_to_label, freqs = None, None, None, None
                model = None

                # read in ALLLLL the files
                gu_train, label_to_id, id_to_label, freqs = load_data(f"data/splits/gu-lp_c_train.conllulex", tokenizer)
                en_train, label_to_id, id_to_label, freqs = load_data(f"data/splits/en-lp_c_train.conllulex", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label)
                en_train2, label_to_id, id_to_label, freqs = load_data(f"data/splits/en-streusle_train.conllulex", tokenizer, label_to_id=label_to_id, id_to_label=id_to_label)
                zh_train, label_to_id, id_to_label, freqs = load_data(f"data/splits/zh-lp_c_train.conllulex", tokenizer,
                                                                      label_to_id=label_to_id, id_to_label=id_to_label)
                jp_train, label_to_id, id_to_label, freqs = load_data(f"data/splits/jp-lp_c_train.conllulex", tokenizer,
                                                                      label_to_id=label_to_id, id_to_label=id_to_label)

                hi_train, label_to_id, id_to_label, freqs = load_data(f"data/splits/hi-lp_c_train.conllulex", tokenizer,
                                                                      label_to_id=label_to_id, id_to_label=id_to_label)

                zh_dev, _, _, _ = load_data(f"data/splits/zh-lp_c_dev.conllulex", tokenizer, label_to_id=label_to_id,
                                            id_to_label=id_to_label)
                zh_test, _, _, _ = load_data(f"data/splits/zh-lp_c_test.conllulex", tokenizer, label_to_id=label_to_id,
                                             id_to_label=id_to_label)

                en_dev, _, _, _ = load_data(f"data/splits/en-lp_c_dev.conllulex", tokenizer, label_to_id=label_to_id,
                                            id_to_label=id_to_label)
                en_test, _, _, _ = load_data(f"data/splits/en-lp_c_test.conllulex", tokenizer, label_to_id=label_to_id,
                                             id_to_label=id_to_label)

                en_dev2, _, _, _ = load_data(f"data/splits/en-streusle_dev.conllulex", tokenizer,
                                             label_to_id=label_to_id, id_to_label=id_to_label)
                en_test2, _, _, _ = load_data(f"data/splits/en-streusle_test.conllulex", tokenizer,
                                              label_to_id=label_to_id, id_to_label=id_to_label)

                jp_dev, _, _, _ = load_data(f"data/splits/jp-lp_c_dev.conllulex", tokenizer, label_to_id=label_to_id,
                                            id_to_label=id_to_label)
                jp_test, _, _, _ = load_data(f"data/splits/jp-lp_c_test.conllulex", tokenizer, label_to_id=label_to_id,
                                             id_to_label=id_to_label)

                hi_dev, _, _, _ = load_data(f"data/splits/hi-lp_c_dev.conllulex", tokenizer, label_to_id=label_to_id,
                                            id_to_label=id_to_label)
                hi_test, _, _, _ = load_data(f"data/splits/hi-lp_c_test.conllulex", tokenizer, label_to_id=label_to_id,
                                             id_to_label=id_to_label)

                gu_dev, _, _, _ = load_data(f"data/splits/gu-lp_c_dev.conllulex", tokenizer, label_to_id=label_to_id,
                                            id_to_label=id_to_label)
                gu_test, _, _, _ = load_data(f"data/splits/gu-lp_c_test.conllulex", tokenizer, label_to_id=label_to_id,
                                             id_to_label=id_to_label)

                train_dataset = gu_train + en_train + en_train2 + zh_train + jp_train + hi_train
                eval_dataset = gu_dev + en_dev + en_dev2 + zh_dev + jp_dev + hi_dev
                test_dataset = gu_test + en_test + en_test2 + zh_test + jp_test + hi_test

                model = AutoModelForTokenClassification.from_pretrained(
                    config.model_name,
                    num_labels=len(label_to_id),
                    id2label=id_to_label,
                    label2id=label_to_id,
                )

                # random.seed(42)
                # random.shuffle(train_dataset)
                # random.seed(42)
                # random.shuffle(eval_dataset)
                # random.seed(42)
                # random.shuffle(test_dataset)



            else:
                # this is most simple case: 1 file, split it into train + eval
                train_dataset = data[len(data) // 5:]
                eval_dataset = data[:len(data) // 5]

    #if you supply a test file separately, you will test on that, and train on training data
    else:
        #if you supply extra data, add that into training too
        if config.extra_file:
            data = extra_data + data #upsample gold data to account for disparity in sizes

        if config.extra_dev_file:
            orig_dev_data = dev_data
            dev_data = extra_dev_data + dev_data

        if config.extra_test_file:
            orig_test_data = test_data
            test_data = extra_test_data + test_data

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

    if config.multilingual:
        gu_results = trainer.predict(gu_test)
        gu_metrics = {"gu_" + k: v for (k, v) in gu_results.metrics.items()}
        wandb.log(gu_metrics)

        en_results = trainer.predict(en_test2)
        en_metrics = {"en_" + k: v for (k, v) in en_results.metrics.items()}
        wandb.log(en_metrics)

        zh_results = trainer.predict(zh_test)
        zh_metrics = {"zh_" + k: v for (k, v) in zh_results.metrics.items()}
        wandb.log(zh_metrics)

        hi_results = trainer.predict(hi_test)
        hi_metrics = {"hi_" + k: v for (k, v) in hi_results.metrics.items()}
        wandb.log(hi_metrics)
        
        jp_results = trainer.predict(jp_test)
        jp_metrics = {"jp_" + k: v for (k, v) in jp_results.metrics.items()}
        wandb.log(jp_metrics)

    if config.extra_test_file:
        #if two test files, compute metrics on each one of them
        lang1 = config.test_file.split("-")[0]
        lang2 = config.extra_test_file.split("-")[0]

        l1_results = trainer.predict(orig_test_data)
        l1_metrics = {lang1 + "_" + k: v for (k, v) in l1_results.metrics.items()}
        wandb.log(l1_metrics)

        l2_results = trainer.predict(extra_test_data)
        l2_metrics = {lang2 + "_" + k: v for (k, v) in l2_results.metrics.items()}
        wandb.log(l2_metrics)


    #use this if you want to save model weights

    if best_metric is None or best_f1 > best_metric:
        with open(".models/best_model/" + lang + "/metric/f1.txt", "w") as f:
            f.write(str(best_f1))

        for f in glob.glob(".models/best_model/" + lang + "/model/.*"):
            os.remove(f)

        model_path = ".models/best_model/" + lang + "/model/"
        trainer.save_model(model_path)





    wandb.finish()

    # #remove model checkpoints, already saved the best one
    # shutil.rmtree("./models/")




def hyper_sweep(args):
    """defines bounds for hyperparameter sweep. Overwrites some arguments."""

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval/f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-7,
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
            'extra_dev_file': {
                "value": args.extra_dev_file
            },
            'extra_test_file': {
                "value": args.extra_test_file
            },
            'multilingual': {
                "value": args.multilingual
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

    wandb.agent(sweep_id, train2, count=75)

BEST_HYPERS = {
    "en": {
        "mono":
            {
                "xlm-roberta-large": {
                    "batch_size": 24,
                    "lr": 4.7e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 28,
                    "weight_decay": .1
                },
                "roberta-large": {
                    "batch_size": 16,
                    "lr": 4.9e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 390,
                    "weight_decay": .1
                }
            }
    },
    "zh":{
        "mono":
            {
                "xlm-roberta-large": {
                    "batch_size": 16,
                    "lr": 6.4e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 457,
                    "weight_decay": 0
                },
                "bert-base-chinese": {
                    "batch_size": 24,
                    "lr": 5.5e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 49,
                    "weight_decay": .1
                }
            },
        "add_en":
            {
                "xlm-roberta-large": {
                    "batch_size": 8,
                    "lr": 5.7e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 225,
                    "weight_decay": 0
                }
        }
    },
    "gu":{
       "mono":{
           "xlm-roberta-large":{
               "batch_size": 16,
               "lr": 5.0e-5,
               "lr_scheduler": "linear",
               "warmup_steps": 244,
               "weight_decay": .1
           },
           "muRIL-large":{
               "batch_size": 4,
               "lr": 3.3e-5,
               "lr_scheduler": "linear",
               "warmup_steps": 296,
               "weight_decay": .1
           }
       },
        "add_en":{
            "xlm-roberta-large": {
                "batch_size": 16,
                "lr": 3.8e-5,
                "lr_scheduler": "linear",
                "warmup_steps": 65,
                "weight_decay": .01
            }
        },
        "add_hi":{
            "xlm-roberta-large": {
                "batch_size": 24,
                "lr": 1.9e-5,
                "lr_scheduler": "cosine",
                "warmup_steps": 82,
                "weight_decay": .1
            },
        }
    },
    "hi":{
        "mono":{
            "xlm-roberta-large": {
                "batch_size": 24,
                "lr": 7.3e-5,
                "lr_scheduler": "linear",
                "warmup_steps": 442,
                "weight_decay": .1
            },
            "muRIL-large": {
                "batch_size": 24,
                "lr": 4.0e-5,
                "lr_scheduler": "linear",
                "warmup_steps": 18,
                "weight_decay": .1
            }
        },
        "add_en":{
            "xlm-roberta-large": {
                "batch_size": 24,
                "lr": 1.6e-5,
                "lr_scheduler": "linear",
                "warmup_steps": 312,
                "weight_decay": .1
            }
        },
        "add_gu":{
            "xlm-roberta-large": {
                "batch_size": 24,
                "lr": 1.9e-5,
                "lr_scheduler": "cosine",
                "warmup_steps": 82,
                "weight_decay": .1
            }
        }
    },
    "jp":{
        "mono":{
            "xlm-roberta-large": {
                "batch_size": 8,
                "lr": 4.4e-5,
                "lr_scheduler": "linear",
                "warmup_steps": 113,
                "weight_decay": .01
            },
            "nlp-waseda/roberta-large-japanese": {
                "batch_size": 16,
                "lr": 1.0e-4,
                "lr_scheduler": "cosine",
                "warmup_steps": 325,
                "weight_decay": .1
            }
        },
        "add_en":{
            "xlm-roberta-large": {
                "batch_size": 8,
                "lr": 1.9e-5,
                "lr_scheduler": "cosine",
                "warmup_steps": 469,
                "weight_decay": .1
            }
        }
    },
    "all":
        {
            "all":{
                "xlm-roberta-large": {
                    "batch_size": 24,
                    "lr": 2.5e-5,
                    "lr_scheduler": "linear",
                    "warmup_steps": 419,
                    "weight_decay": 0
                }
            }
        }

}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--loss_fn", type=str, default=None) #overwritten by wandb sweep
    parser.add_argument("--file", type=str, default="hi-lp_c_train.conllulex")
    parser.add_argument("--learning_rate", type=float, default=1.6e-5) #overwritten by wandb sweep
    parser.add_argument("--batch_size", type=int, default=24) #overwritten by wandb sweep
    parser.add_argument("--epochs", type=int, default=10) #overwritten by wandb sweep
    parser.add_argument("--weight_decay", type=float, default=0.1) #overwritten by wandb sweep
    parser.add_argument("--freeze", action="store_true") #overwritten by wandb sweep
    parser.add_argument("--warmup_steps", type=int, default=312)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--test_file", type=str, default="hi-lp_c_test.conllulex", help="Need to put file for test split")
    parser.add_argument("--dev_file", type=str, default="hi-lp_c_dev.conllulex", help="Need to put file for dev split")
    parser.add_argument("--extra_file", type=str, default=None, help="If you want to add an extra file to add more data during the fine-tuning stage. Evaluation is still only perfomed on the original file test split.")
    parser.add_argument("--extra_dev_file", type=str, default=None, help="Add in an extra dev file (another lang for data sharing)")
    parser.add_argument("--extra_test_file", type=str, default=None, help="Add in an extra test file (another lang for data sharing)")
    parser.add_argument("--lang", type=str, default="en", help="Specify the language that you want to load the trained model for")
    parser.add_argument("--multilingual", action="store_true", help="If supplying an extra lang file, put true to include that language in eval. Otherwise it will only test on original lang")
    parser.add_argument("--do_sweep", action="store_true")  #this flag makes the sweep happen versus an individual training run
    parser.add_argument("--eval_only", action="store_true")  # this flag makes the sweep happen versus an individual training run
    parser.add_argument("--predict_only", action="store_true")  # this flag makes the sweep happen versus an individual training run
    parser.add_argument("--use_best_hypers", action="store_true", help="Use this to automatically use the best hyperparameters for a language and extra language. Overwrites other arguments.")
    parser.add_argument("--extra_lang", type=str, default=None, help="The language that you want to use for supplemental data. Overwrites extra file argument.")





    args = parser.parse_args()

    if args.do_sweep:
        hyper_sweep(args)

    else:



        if args.eval_only:
            load_trained_model(args.lang, args.file, args.dev_file, args.test_file, do_eval=True)

        elif args.predict_only:
            load_trained_model(args.lang, args.file, args.dev_file, args.test_file, do_eval=False)

        else:
            if args.use_best_hypers:
                model_name = args.model_name
                if args.extra_lang:
                    setting = "add_" + args.extra_lang
                else:
                    setting = "mono"

                lr = BEST_HYPERS[args.lang][setting][model_name]["lr"]
                epochs = 10
                decay = BEST_HYPERS[args.lang][setting][model_name]["weight_decay"]
                warmup = BEST_HYPERS[args.lang][setting][model_name]["warmup_steps"]
                batch_size = BEST_HYPERS[args.lang][setting][model_name]["batch_size"]
                scheduler = BEST_HYPERS[args.lang][setting][model_name]["lr_scheduler"]

                train(args.model_name, args.file, lr, batch_size, epochs,
                      decay, False, args.test_file, args.dev_file, args.extra_file,
                      warmup_steps=warmup, lr_scheduler=scheduler, number=0)

            else:
                for number in range(20,200):
                    train(args.model_name, args.file, args.learning_rate, args.batch_size, args.epochs,
                          args.weight_decay, args.freeze, args.test_file, args.dev_file, args.extra_file, warmup_steps=args.warmup_steps, lr_scheduler=args.lr_scheduler, number=number)


        # model_name: str,  # need - added
        # file: str,  # need - added
        # learning_rate: float,  # don't need
        # batch_size: int,  # don't need
        # epochs: int,  # don't need
        # weight_decay: float,  # don't need
        # freeze: bool,  # need - added
        # test_file: str,  # need
        # extra_file: str,  # need
        # multilingual = False,  # need
        # loss_fn = None,  # need - added
        # do_sweep = False


if __name__ == "__main__":
    main()
