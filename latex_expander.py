# 1) go though the example directory and for each file read it
import os
import re

BEGIN_DOCUMENT_LATEX = "\\begin{document}"

unexpanded_dir = "data/example"
# s = re.compile(r'?P<maro_name>\\newcommand{?\\[a-zA-Z0-9]+}?')
s = re.compile(r'\\newcommand{?\\[a-zA-Z0-9]+}?')

for subdir, dirs, files in os.walk(unexpanded_dir):
    for f in files:
        delete_intermediate_files()

        # the algorithm
        with open(f, 'r') as unexpanded:
            # TODO:...
            buildup_string = ""
            l_count, r_count = 0, 0
            f_name = os.path.basename(f.name)
            building = False

            # the copy file for copying all non-macros
            # the sty file for the current document/file
            with open(f_name+"_copy", "w+") as expanded:
                with open(f_name + ".sty", "w+") as sty:
                    for line in unexpanded:
                        if is_beginning_of_document(line):
                            break
                        else:
                            # if no match, just write to copy of file
                            if not building:
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
                                    sty.write(buildup_string)
                                    buildup_string = ""
                                    l_count, r_count = 0, 0











        run_demacro()
        put_clean_latex_in_proper_directory()
