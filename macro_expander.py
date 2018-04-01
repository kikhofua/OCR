import os


def expand():
    directory = "../data/faux_latex"
    for filename in os.listdir(directory):
        swap_file = os.path.join(directory, os.path.splitext(filename)[0]+"_swap.tex")
        with open(swap_file, "w+") as fswap:
            for line in open(os.path.join(directory, filename)):
                if "\\def" in line:
                    line = line.replace("\\def", "\\newcommand")
                    fswap.write(line)
                else:
                    fswap.write(line)

    return


if __name__ == "__main__":
    expand()
