import shutil
import os


GOOD_LATEX_DIR = "data/good_latex"
BAD_LATEX_DIR = "data/bad_latex_files"


for subdir, dirs, files in os.walk(GOOD_LATEX_DIR):
    for f_name in files:
        f_path = os.path.join(GOOD_LATEX_DIR, f_name)
        with open(f_path, 'r+') as unexpanded:
            try:
                for line in unexpanded:
                    pass
            except Exception:
                base_name = os.path.basename(f_path)
                print("Found bad file! {}".format(base_name))
                src_path = f_path
                dst_path = os.path.join(BAD_LATEX_DIR, base_name)
                shutil.move(src_path, dst_path)
