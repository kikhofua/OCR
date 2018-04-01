from queue import Queue
import os, re


FULL_DOCUMENTS_DIR = "data/expanded_latex/"
SNIPPETS_DIR = "data/snippets/"

SLIDING_WINDOW = 5

BEGIN_LATEX_DOC = "\\begin{document}"
END_LATEX_DOC = "\\end{document}"


begin_block = re.compile(r'\\begin{(?P<block_name>[a-zA-Z0-9]+)}')
entire_block = re.compile(r'^\\begin{(?P<block>[a-zA-Z0-9]+)}.*\\end{(?P=block)}$', re.DOTALL)

def create_new_snippet_from_queue(lines, snippet_number):
    snippet_path = os.path.join(SNIPPETS_DIR, "sn_{}".format(snippet_number))
    with open(snippet_path, 'w+') as snippet:
        snippet.write("\\documentstyle[12pt]{article}\n")
        snippet.write("\\begin{document}\n")
        for l in lines:
            snippet.write("{}\n".format(l))
        snippet.write("\\end{document}\n")


for subdir, dirs, files in os.walk(FULL_DOCUMENTS_DIR):
    snippet_counter = 0
    for f_name in files:
        print(f_name)
        in_document_body = False
        lines_queue = Queue(maxsize=SLIDING_WINDOW)
        f_path = os.path.join(FULL_DOCUMENTS_DIR, f_name)
        with open(f_path, 'r') as document:
            building_block = False
            multi_line_builder = ""
            for line in document:
                if not in_document_body:
                    if BEGIN_LATEX_DOC in line:
                        in_document_body = True
                else:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    if END_LATEX_DOC in line:
                        break
                    if not building_block:
                        if re.search(begin_block, line):
                            building_block = True
                    if building_block:
                        match = re.search(entire_block, multi_line_builder)
                        if match:
                            line = multi_line_builder
                            # print("{}$$$$$$$$$$$$$$${}$$$$$$$$$$$$$$$$$$$$$$$".format(multi_line_builder, snippet_counter))
                            multi_line_builder = ""
                            building_block = False
                        else:
                            multi_line_builder += "{}\n".format(line)
                            continue
                    lines_queue.put(line)
                    if lines_queue.full():
                        create_new_snippet_from_queue(list(lines_queue.queue), snippet_counter)
                        snippet_counter += 1
                        lines_queue.get()
