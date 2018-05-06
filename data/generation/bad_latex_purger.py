import os, sys
# import subprocess as sub
# # 1) need to be able to specify a file to start at
# # 2) need to be able to catch system exit 1 errors
# # 3) need to be able to be able to time out after 10 seconds
#
# datadir = "data/latex"
# pdfsdir = "data/PDFs"
# latex_command = "pdflatex -interaction=batchmode -output-directory=" + pdfsdir + " {}"
#
#
# files_list = []
# for subdir, dirs, files in os.walk(datadir):
#     for file in files:
#         # print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file
#         files_list.append(filepath)
#
# # start_file = datadir + "\\1993\\9309039"
# # start_index = files_list.index(start_file)
# # files_list = files_list[start_index:]
#
#
# # identify valid latex files
# bad_latex_files = []
# for file in files_list:
#     try:
#         print("Processing:" + file + " ###############################")
#         sub.run(["pdflatex", "-interaction=batchmode", "-output-directory=data/PDFs", file], timeout=5)
#     except sub.TimeoutExpired:
#         print("DOCUMENT TOOK TOO LONG TO COMPILE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#         bad_latex_files.append(file)
#     finally:
#         print("\n\n\n")
#         # sys.stdout.flush()
#
#
# subdir_prefixes = \
#     {"00": "2000",
#      "01": "2001",
#      "02": "2002",
#      "03": "2003",
#      "92": "1992",
#      "93": "1993",
#      "94": "1994",
#      "95": "1995",
#      "96": "1996",
#      "97": "1997",
#      "98": "1998",
#      "99": "1999"}
# with open('data/good_files.txt', 'a') as good_latex_file:
#     for subdir, dirs, files in os.walk(pdfsdir):
#         for file in files:
#             if file.endswith(".pdf"):
#                 subdir_pref = subdir_prefixes[file[0:2]]
#                 file_path = os.path.join("data", "latex", subdir_pref, file)
#                 good_latex_file.write("{}\n".format(file_path))
# import os
# from shutil import copy
# dest_dir = os.path.join("data", "good_latex")
# with open('data/good_files.txt', 'r') as good_latex_file:
#     for file_name in good_latex_file:
#         copy(file_name[:-5], dest_dir)