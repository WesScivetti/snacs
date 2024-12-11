import conllu
import argparse


def print_desc_stats(infile):
    """
    prints descriptive stats for a conllulex file
    """

    conllulex = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps',
                 'misc', 'smwe', 'lexcat', 'lexlemma', 'ss', 'ss2', 'wmwe', 'wcat', 'wlemma', 'lextag']


    with open(infile, "r") as fin:
        sentence_count = 0
        token_count = 0
        snacs_examples = []
        tag_examples = []
        scene_examples = []
        func_examples = []

        for sentence in conllu.parse_incr(fin, fields=conllulex):
            sentence_count += 1
            for word in sentence:
                token_count += 1
                if "p." in word["ss"]:
                    # print(word["lextag"])
                    snacs_examples.append(word["form"])
                    tag_examples.append(word["ss"] + "-" + word["ss2"])
                    scene_examples.append(word["ss"])
                    func_examples.append(word["ss2"])

        print("# of SENTENCES:", sentence_count)
        print("# of TOKENS:", token_count)
        print("# of SNACS TOKENS:", len(snacs_examples))
        print("# of SNACS TYPES:", len(list(set(snacs_examples))))
        print("# of UNIQUE CONSTRUALS:", len(list(set(tag_examples))))
        print("# of UNIQUE SCENE ROLES:", len(list(set(scene_examples))))
        # print(list(set(scene_examples)))
        print("# of UNIQUE CODING FUNCTIONS:", len(list(set(func_examples))))

    #
    # with open(train_name, "w") as fout_train:
    #     fout_train.writelines([sentence.serialize() + "\n" for sentence in train_sents])
    #
    # with open(dev_name, "w") as fout_dev:
    #     fout_dev.writelines([sentence.serialize() + "\n" for sentence in dev_sents])
    #
    # with open(test_name, "w") as fout_test:
    #     fout_test.writelines([sentence.serialize() + "\n" for sentence in test_sents])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="en-lp_c.conllulex")
    args = parser.parse_args()
    print_desc_stats(args.file)

