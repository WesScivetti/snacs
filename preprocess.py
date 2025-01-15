import stanza
import conllu
from argparse import ArgumentParser
from conllu import parse_incr

def preprocess_conllu(input_file):
    outfile_name = "".join(input_file.split(".")[:-1]) + "_preprocessed.conllulex"

    with open(outfile_name, "w") as outf:
        with open(input_file, "r") as inf:
            for line in inf:
                if len(line) == 1:
                    outf.write(line)
                elif line.startswith("#"):
                    outf.write(line)
                else:
                    l = line.rstrip("\n").split("\t")
                    # print(len(l))
                    # print(l)
                    extra_stuff = ["_", "_", "_", "_", "_", "_", "_", "_", "-"]
                    newl = l + extra_stuff
                    newline = "\t".join(newl) + "\n"
                    outf.write(newline)

import stanza
from stanza.utils.conll import CoNLL

def preprocess_text_plain(input_file, lang="en"):
    # Initialize the Stanza pipeline for the desired language
    stanza.download(lang)
    nlp = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)

    # Process the input text with Stanza to generate CoNLL-U formatted output

    with open(input_file, "r") as inf:
        text = inf.read()
        doc = nlp(text)

    outfile_conllu = "".join(input_file.split(".")[:-1]) + ".conllu"

    outfile_lex = "".join(input_file.split(".")[:-1]) + "_preprocessed.conllulex"

    CoNLL.write_doc2conll(doc, outfile_conllu)

    preprocess_conllu(outfile_conllu)

    # outfile_name = "preprocessed_output.conllulex"
    #
    # with open(outfile_name, "w") as outf:
    #     for line in conllu_lines:
    #         if len(line) == 0:
    #             outf.write("\n")
    #         elif line.startswith("#"):
    #             outf.write(line + "\n")
    #         else:
    #             l = line.split("\t")
    #             # Adding extra columns
    #             extra_stuff = ["_", "_", "_", "_", "_", "_", "_", "_", "-"]
    #             newl = l + extra_stuff
    #             newline = "\t".join(newl) + "\n"
    #             outf.write(newline)

    #print(f"Preprocessed CoNLL-U file with extra columns saved to {outfile_name}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--input_format", help="Please specify the input format of the file to preprocess. If the file is in conllu, type 'conllu'. If plain text, type 'plain'. ")
    parser.add_argument("--lang", help="specify the language (inportant for POS tagging/identifying target adpositions)")
    args = parser.parse_args()
    if args.input_format == "conllu":
        preprocess_conllu(args.input_file)
    if args.input_format == "plain":
        preprocess_text_plain(args.input_file, args.lang)