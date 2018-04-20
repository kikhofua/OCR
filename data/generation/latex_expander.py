from data.generation import expander_utils as expander
import os
import re


STY_FILE = "intermediate-private.sty"
STY_HEADER = "\\usepackage{intermediate-private}"
INTERMEDIATE_LATEX = os.path.join(expander.INTERMEDIATE_LATEX_DIR, expander.INTERMEDIATE_LATEX_FILE)
INTERMEDIATE_STY = os.path.join(expander.INTERMEDIATE_LATEX_DIR, STY_FILE)
UNEXPANDED_DIR = "data/example"



s = re.compile(r'\\(newcommand|def){?\\[a-zA-Z0-9]+}?')

for subdir, dirs, files in os.walk(UNEXPANDED_DIR):
    for f_name in files:
        expander.delete_intermediate_files()

        # the algorithm
        f_path = os.path.join(UNEXPANDED_DIR, f_name)
        with open(f_path, 'r') as unexpanded:
            buildup_string = ""
            l_count, r_count = 0, 0
            building = False
            in_body = False

            # the copy file for copying all non-macros
            # the sty file for the current document/file
            with open(INTERMEDIATE_LATEX, "w+") as expanded:
                expanded.write(STY_HEADER)

                with open(INTERMEDIATE_STY, "w+") as sty:
                    for line in unexpanded:
                        if expander.is_beginning_of_document(line):
                            in_body = True

                        if not in_body:
                            if not building:
                                # if no match, just write to copy of file
                                if s.match(line) is None:
                                    expanded.write(line)
                                else:
                                    building = True

                            # line contains a macro so write to sty
                            if building:
                                buildup_string += line
                                l_count += line.count("{")
                                r_count += line.count("}")
                                if l_count == r_count:
                                    building = False
                                    if "#" in buildup_string or "@" in buildup_string:
                                        expanded.write(buildup_string)
                                    else:
                                        if buildup_string.count("\\newcommand") > 1 or buildup_string.count("\\def") > 1:
                                            expanded.write(buildup_string)
                                        else:
                                            buildup_string = buildup_string.replace("\\def", "\\newcommand")
                                            sty.write(buildup_string)
                                    buildup_string = ""
                                    l_count, r_count = 0, 0

                        if in_body:
                            expanded.write(line)
        expander.run_demacro()
        expander.put_clean_latex_in_proper_directory(f_path)

