from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from load_data import tokenize_and_align
import numpy as np
import evaluate
import random
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seqeval = evaluate.load("seqeval")

def load_data(file: str, tokenizer: AutoTokenizer):
    res = tokenize_and_align(file, tokenizer)

    # make label-id mapping
    label_to_id = {"None": -100}
    id_to_label = {-100: "None"}
    for sent, mask, label in res:
        for i in range(len(sent)):
            if mask[i]:
                if label[i] not in label_to_id:
                    id = len(label_to_id)
                    label_to_id[label[i]] = id
                    id_to_label[id] = label[i]

    res2 = []

    # add sos and eos, convert labels to ids
    sos_eos = tokenizer("")['input_ids']
    for sent, mask, label in res:
        if len(sos_eos) == 2:
            sent = [sos_eos[0]] + sent + [sos_eos[1]]
            mask = [0] + mask + [0]
            label = ["None"] + label + ["None"]
        label = [label_to_id[x] for x in label]
        res2.append({
            'input_ids': sent,
            'mask': mask,
            'labels': label
        })
    
    print(f"{len(label_to_id)} labels.")
    random.shuffle(res2)
    
    return res2, label_to_id, id_to_label

def compute_metrics(p, id_to_label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train(model_name: str, file: str, learning_rate: float, batch_size: int, epochs: int, weight_decay: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data, label_to_id, id_to_label = load_data(file, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_to_id), id2label=id_to_label, label2id=label_to_id
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data[len(data) // 5:],
        eval_dataset=data[:len(data) // 5],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id_to_label),
    )

    trainer.train()

def main():
    # argparse to get model name and data file, learning rate, batch size, epochs, weight decay
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--file", type=str, default="data/en-lp.conllulex")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    train(**vars(args))

if __name__ == "__main__":
    main()