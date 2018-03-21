import os

# 1) need to be able to specify a file to start at
# 2) need to be able to catch system exit 1 errors
# 3) need to be able to be able to time out after 10 seconds




datadir = "data/latex"
pdfsdir = "data/PDFs"
latex_command = "pdflatex -interaction=batchmode -output-directory=" + pdfsdir + " {}"
files_list = []
for subdir, dirs, files in os.walk(datadir):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        files_list.append(filepath)

start_file = datadir + "\\1993\\9309039"
start_index = files_list.index(start_file)
files_list = files_list[start_index:]

for file in files_list:
    os.system(latex_command.format(file))
    print(file + "\n\n\n")
