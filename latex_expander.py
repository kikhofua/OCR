import expander_utils as expander
import os
import re



unexpanded_dir = "data/example"
s = re.compile(r'\\newcommand{?\\[a-zA-Z0-9]+}?')

for subdir, dirs, files in os.walk(unexpanded_dir):
    for f in files:
        expander.delete_intermediate_files()
        # the algorithm
        with open(f, 'r') as unexpanded:
            buildup_string = ""
            l_count, r_count = 0, 0
            f_name = os.path.basename(f.name)
            building = False
            in_body = False

            # the copy file for copying all non-macros
            # the sty file for the current document/file
            with open(f_name+"_copy", "w+") as expanded:
                with open(f_name + ".sty", "w+") as sty:
                    for line in unexpanded:
                        if expander.is_beginning_of_document(line):
                            in_body = True

                        if not in_body:
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

                        if in_body:
                            expanded.write(line)
        expander.run_demacro()
        expander.put_clean_latex_in_proper_directory(file.name)

