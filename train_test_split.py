import conllu
import argparse


def train_test_split(infile, output_dir):
    """
    split litte prince data based on the chapter splits of dev chapters [1, 10, 20] and test chapters [7, 17, 27]
    """

    conllulex = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps',
                 'misc', 'smwe', 'lexcat', 'lexlemma', 'ss', 'ss2', 'wmwe', 'wcat', 'wlemma', 'lextag']

    dev_chapters = [1, 10, 20]
    test_chapters = [7, 17, 27]
    orig_fname = infile.split("/")[-1].split(".")[0]
    train_name = output_dir + orig_fname + "_train.conllulex"
    dev_name = output_dir + orig_fname + "_dev.conllulex"
    test_name = output_dir + orig_fname + "_test.conllulex"

    train_sents = []
    dev_sents = []
    test_sents = []


    with open(infile, "r") as fin:
        for sentence in conllu.parse_incr(fin, fields=conllulex):
            chapter = int(sentence.metadata["chapter"])

            if chapter in dev_chapters:
                dev_sents.append(sentence)

            elif chapter in test_chapters:
                test_sents.append(sentence)

            else:
                train_sents.append(sentence)

    with open(train_name, "w") as fout_train:
        fout_train.writelines([sentence.serialize() + "\n" for sentence in train_sents])

    with open(dev_name, "w") as fout_dev:
        fout_dev.writelines([sentence.serialize() + "\n" for sentence in dev_sents])

    with open(test_name, "w") as fout_test:
        fout_test.writelines([sentence.serialize() + "\n" for sentence in test_sents])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="en-lp_c.conllulex")
    parser.add_argument("--output_dir", type=str, default="data/splits/")
    args = parser.parse_args()
    train_test_split(args.file, args.output_dir)

