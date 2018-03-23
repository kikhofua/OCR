import os, sys
from utils import timeout
# 1) need to be able to specify a file to start at
# 2) need to be able to catch system exit 1 errors
# 3) need to be able to be able to time out after 10 seconds

datadir = "data/latex"
pdfsdir = "data/PDFs"


@timeout(5, "Document took too long and error wasn't caught")
def compile_latex_file(file_name):
    latex_command = "pdflatex -interaction=batchmode -output-directory=" + pdfsdir + " {}"
    os.system(latex_command.format(file_name))


files_list = []
for subdir, dirs, files in os.walk(datadir):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        files_list.append(filepath)

# start_file = datadir + "\\1993\\9309039"
# start_index = files_list.index(start_file)
# files_list = files_list[start_index:]


# identify valid latex files
bad_latex_files = []
for file in files_list:
    try:
        compile_latex_file(file)
        print(file + "\n\n\n")
    except TimeoutError:
        print("\n\n\nDOCUMENT TOOK TOO LONG TO COMPILE\n\n\n")
        bad_latex_files.append(file)
    sys.stdout.flush()


with open('/data/bad_files.txt', 'a') as bad_latex_names:
    for bad_latex in bad_latex_files:
        bad_latex_names.write("{}\n".format(bad_latex))
print(bad_latex_files)

