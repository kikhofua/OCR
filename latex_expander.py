import expander_utils as expdr
import os



unexpanded_dir = "data/example"
for subdir, dirs, files in os.walk(unexpanded_dir):
    for file in files:
        expdr.delete_intermediate_files()

        # the algorithm
        with open(file, 'r') as unexpanded:
            # TODO:...
            for line in unexpanded:
                if expdr.is_beginning_of_document(line):
                    break
                else:
                    # TODO:....


        expdr.run_demacro()
        expdr.put_clean_latex_in_proper_directory(file.name)

