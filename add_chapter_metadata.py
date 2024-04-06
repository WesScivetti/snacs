import re

target_files = ["en-lp.conllulex", "gu-lp.conllulex", "hi-lp.conllulex", "lpp_jp.conllulex", "zh-lp.conllulex"]
data_dir = "./data/"
for i, filename in enumerate(target_files):
    if i == 0:
        #ENGLISH
        chap_num = 0
        out_name = data_dir + "en-lp_c.conllulex"
        with open(out_name, "w") as outf:
            with open(data_dir + filename, "r") as inf:
                for line in inf:
                    if line.startswith("# text"):
                        if re.search("Chapter", line):
                            chap_num += 1
                            outf.write("# chapter = " + str(chap_num) + "\n")
                    outf.write(line)

    if i == 1:
        #GUJARATI
        out_name = data_dir + "gu-lp_c.conllulex"
        with open(out_name, "w") as outf:
            with open(data_dir + filename, "r") as inf:
                for line in inf:
                    if line.startswith("# sent_id"):
                        chap = line.split("_")[3]
                        outf.write(line)
                        outf.write("# chapter = " + str(chap) + "\n")
                    else:
                        outf.write(line)

    if i == 2:
        #HINDI
        out_name = data_dir + "hi-lp_c.conllulex"
        with open(out_name, "w") as outf:
            with open(data_dir + filename, "r") as inf:
                for line in inf:
                    if line.startswith("# sent_id"):
                        chap = line.split("_")[3].split("-")[0]
                        outf.write(line)
                        outf.write("# chapter = " + str(chap) + "\n")
                    else:
                        outf.write(line)

    if i == 3:
        #Japanese
        out_name = data_dir + "jp-lp_c.conllulex"
        with open(out_name, "w") as outf:
            with open(data_dir + filename, "r") as inf:
                for line in inf:
                    if line.startswith("# sent_id"):
                        chap = line.split(".")[1]
                        outf.write(line)
                        outf.write("# chapter = " + str(chap) + "\n")
                    else:
                        outf.write(line)

    if i == 4:
        #Chinese
        out_name = data_dir + "zh-lp_c.conllulex"
        with open(out_name, "w") as outf:
            with open(data_dir + filename, "r") as inf:
                for line in inf:
                    if line.startswith("# newdoc_id"):
                        chap = line.split("_")[4].split("-")[1]
                        outf.write(line)
                        hold = True

                    else:
                        outf.write(line)
                        if hold == True:
                            outf.write("# chapter = " + str(int(chap)) + "\n")
                            hold = False


