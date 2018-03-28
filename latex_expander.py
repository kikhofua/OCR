# 1) go though the example directory and for each file read it
import os

BEGIN_DOCUMENT_LATEX = "\\begin{document}"

unexpanded_dir = "data/example"
for subdir, dirs, files in os.walk(unexpanded_dir):
    for file in files:
        delete_intermediate_files()

        # the algorithm
        with open(file, 'r') as unexpanded:
            # TODO:...
            for line in unexpanded:
                if is_beginning_of_document(line):
                    break
                else:
                    # TODO:....


        run_demacro()
        put_clean_latex_in_proper_directory()
