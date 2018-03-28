import os
import shutil
import subprocess as sub


INTERMEDIATE_LATEX_FILE = "intermeciate.tex"
CLEANED_LATEX_FILE = "intermediate-clean.tex"
INTERMEDIATE_LATEX_DIR = "data/intermediate/"
FINAL_LATEX_DIR = "data/expanded_latex/"
BEGIN_DOCUMENT_LATEX = "\\begin{document}"


def delete_intermediate_files():
    for the_file in os.listdir(INTERMEDIATE_LATEX_DIR):
        file_path = os.path.join(INTERMEDIATE_LATEX_DIR, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def is_beginning_of_document(line):
    return BEGIN_DOCUMENT_LATEX in line


def run_demacro():
    tex_path = os.path.join(INTERMEDIATE_LATEX_DIR, INTERMEDIATE_LATEX_FILE)
    sub.run(["demacro", tex_path])


def put_clean_latex_in_proper_directory(original_file):
    base_name = os.path.basename(original_file)
    src_path = os.path.join(INTERMEDIATE_LATEX_DIR, CLEANED_LATEX_FILE)
    dst_path = os.path.join(FINAL_LATEX_DIR, base_name)
    shutil.move(src_path, dst_path)
