import conllu
from transformers import AutoTokenizer
import glob
from tqdm import tqdm
from collections import defaultdict
from math import log
import sys

conllulex = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head','deprel', 'deps',
             'misc', 'smwe', 'lexcat', 'lexlemma', 'ss', 'ss2', 'wmwe', 'wcat', 'wlemma', 'lextag']

def tokenize_and_align(
    file: str,
    tokenizer: AutoTokenizer,
    add_space: bool=True,
    hide_non_adp: bool=True,
    o_to_b: bool=True,
    verbose: bool=False
):
    # remember start and end tokens for later
    delimiters = list(tokenizer.encode(''))
    start_token, end_token = (None, None) if len(delimiters) != 2 else (delimiters[0], delimiters[1])
    
    print(file)
    res = []
    with open(file, 'r') as fin:

        dev_chaps = [1, 10, 20]
        test_chaps = [7, 17, 27]

        failed = 0
        for sent in tqdm(conllu.parse_incr(fin, fields=conllulex)):
            text = sent.metadata['text']

            chapter = int(sent.metadata["chapter"])
            if chapter in dev_chaps:
                split = "dev"
            elif chapter in test_chaps:
                split = "test"
            else:
                split = "train"

            tokens, mask, labels, lexlemmas = [], [], [], []
            work = True

            smwe_tags = {}

            for i, token in enumerate(sent):
                # ignore subword tokens (is this a good idea?)
                if not isinstance(token['id'], int): continue

                # ignore non-adposition labels (e.g. in streusle, which has noun/verb labels)
                if hide_non_adp:
                    if not token['ss'].startswith('p.'):
                        token['ss'] = '_'
                        token['ss2'] = '_'

                # generate BIO tag
                # every span starts with a B (BIO-2 style)
                if token['smwe'] != '_':
                    smwe, pos = token['smwe'].split(':')
                    if smwe not in smwe_tags:
                        if token['ss'] != '_':
                            smwe_tags[smwe] = token['ss'] + '-' + token['ss2']
                            token['lextag'] = 'B-' + smwe_tags[smwe]
                        else:
                            token['lextag'] = 'O'
                    else:
                        token['lextag'] = 'I-' + smwe_tags[smwe]
                elif token['ss'] != '_':
                    token['lextag'] = 'B-' + token['ss'] + '-' + token['ss2']
                else:
                    token['lextag'] = 'O'

                # tokenize using LM tokenizer
                tok = None
                if i == 0 or not add_space: tok = tokenizer.encode(token['form'])
                else: tok = tokenizer.encode(' ' + token['form'])

                # remove bos/eos/cls/sep tokens
                tok = [t for t in tok if t not in delimiters]

                # add tag + 'None' for each remaining subword token
                labels.extend([token['lextag']] + ['None' for _ in range(len(tok) - 1)])
                mask.extend([1] + [0 for _ in range(len(tok) - 1)])
                lexlemmas.extend([token['lexlemma']] + ['None' for _ in range(len(tok) - 1)])
                tokens.extend(tok)
            
            # now add bos/cls and eos/sep tokens
            if start_token is not None:
                tokens = [start_token] + tokens
                mask = [0] + mask
                labels = ['None'] + labels
                lexlemmas = [''] + lexlemmas
            if end_token is not None:
                tokens.append(end_token)
                mask.append(0)
                labels.append('None')
                lexlemmas.append('')
            
            # append to result
            if work:
                assert len(tokens) == len(mask) == len(labels)
                res.append([tokens, mask, labels, lexlemmas, split])

    # print out some examples
    if verbose:
        for tokens, mask, labels, lexlemmas in res:
            for j in range(len(tokens)):
                print(f"{tokens[j]:<10} {tokenizer.decode(tokens[j]):<15} {mask[j]:<10} {labels[j]:<30}")
            input()

    # stats
    if failed != 0:
        print(f"Failed to parse {failed} sentences, did {len(res)}.")
    else:
        print(f"Done: {len(res)} sentences.")

    return res

def get_ss_frequencies(res: list):
    """prints out the relative frequencies of each SS, SS2 and lextag after a file has been tokenized and aligned.
    The relative frequencies can be used for weighting a loss function during training. 
    CURRENTLY NOT IN USE"""
    print("STARTING FREQUENCIES")
    inv_freqs = {}
    freqs = defaultdict(lambda: defaultdict(int))

    for tokens, mask, labels, lexlemmas, split in res:

        #filter out masked tokens from labels
        labels = [l for i, l in enumerate(labels) if mask[i] == 1]
        for lab in labels:
            lextag = lab
            l = lab.split("-")
            if len(l) == 1:
                #lextag is "O", so both ss and ss2 are set to "O"
                ss = "O"
                ss2 = "O"
            if len(l) == 3:
                #lextag has B-ss-ss2 form. meaning the lextag only has one ss, meaning ss2 == ss.
                ss = l[1]
                ss2 = l[2]

            freqs["lt"][lextag] += 1
            freqs["ss"][ss] += 1
            freqs["ss2"][ss2] += 1


    total_lt = len(freqs["lt"].keys())
    total_ss = len(freqs["ss"].keys())
    total_ss = len(freqs["ss2"].keys())
    
    return freqs

def inversify_freqs(freqs):
    """basically turns frequencies into log inverse freqs / surprisal
    CURRENTLY NOT IN USE
    """

    inv_freqs = {"lt": {}, "ss": {}, "ss2": {}}
    #gotta populate the inverse frequencies using the frequencies
    for thing in freqs: #thing is either lt, ss, or ss2
        inv_freqs[thing] = {}
        for tag in freqs[thing]:
            # print(thing, tag, freqs[thing][tag], file=sys.stderr)
            inv_freqs[thing][tag] = 1 / log(freqs[thing][tag] + 1)
    return inv_freqs



def main():
    for file in glob.glob("data/en-test.conllulex"):
        print(file)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = tokenize_and_align(file, tokenizer, verbose=False)
        print(dataset[0])

if __name__ == "__main__":
    main()