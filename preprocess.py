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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--input_format", help="Please specify the input format of the file to preprocess. If the file is in conllu, type 'conllu'. If plain text, type 'plain'. ")
    parser.add_argument("--lang", help="specify the language (inportant for POS tagging/identifying target adpositions)")
    args = parser.parse_args()
    preprocess_conllu(args.input_file)