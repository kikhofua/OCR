"""
input : a directory containing files which are latex snippets
output: a directory containing files pdf snippets of those latex snippets
"""

import os

LATEX_SNIPPET_DIR = "data/latex_snippets"
PDF_SNIPPET_DIR = "data/pdf_snippets"
os.chdir("/Users/Kamoya/OCR")

if not os.path.exists(PDF_SNIPPET_DIR):  # create the build directory if not existing
    os.makedirs(PDF_SNIPPET_DIR)


for subdir, dirs, files in os.walk(LATEX_SNIPPET_DIR):
    for f_name in files:
        f_path = os.path.join(LATEX_SNIPPET_DIR, f_name)
        w_path = os.path.join(PDF_SNIPPET_DIR, f_name)
        with open(f_path, 'r+') as latex_snippet:
            with open(w_path, 'w+') as pdf_snippet:
                os.system("pdflatex -interaction=batchmode "
                          "-output-directory={} {}".format(os.path.realpath(PDF_SNIPPET_DIR), f_path))

os.chdir("/Users/Kamoya/OCR/data/pdf_snippets")
list = os.listdir("/Users/Kamoya/OCR/data/pdf_snippets")
for item in list:
        if item.endswith(".aux") or item.endswith(".log") or item.endswith(".tex") or item.endswith(".toc"):
            os.remove(item)





