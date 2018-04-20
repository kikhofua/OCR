import os, re

from queue import Queue


class DataGenerator:



    def __init__(self, source, destination, snippet_size):
        self.doc_dir = source
        self.snippet_dir = destination
        self.snippet_size = snippet_size
        self.snippet_counter = 0

    def generate(self):
        self.snippet_counter = 0
        latex_docs_src = os.fsencode(self.doc_dir)
        for file in os.listdir(latex_docs_src):
            filename = os.fsdecode(file)
            filepath = os.path.join(latex_docs_src, filename)
            # TODO: method that goes through all of the good latex documents and generates normalized text snippets and uniformly sized images of the rendered PDF for the snippet

    def extract_snippets_from_latex_document(self, filepath):
        begin_latex_document = "\\begin{document}"
        end_latex_document = "\\end{document}"
        begin_block = re.compile(r'\\begin{(?P<block_name>[a-zA-Z0-9]+)}')
        entire_block = re.compile(r'^\\begin{(?P<block>[a-zA-Z0-9]+)}.*\\end{(?P=block)}$', re.DOTALL)
        with open(filepath, 'r') as document:
            in_document_body = False
            lines_queue = Queue(maxsize=self.snippet_size)
            building_block = False
            multi_line_builder = ""
            for line in document:
                if not in_document_body:
                    if begin_latex_document in line:
                        in_document_body = True
                else:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    if end_latex_document in line:
                        break
                    if not building_block:
                        if re.search(begin_block, line):
                            building_block = True
                    if building_block:
                        match = re.search(entire_block, multi_line_builder)
                        if match:
                            line = multi_line_builder
                            multi_line_builder = ""
                            building_block = False
                        else:
                            multi_line_builder += "{}\n".format(line)
                            continue
                    lines_queue.put(line)
                    if lines_queue.full():
                        self.create_normalized_snippet_from_lines(list(lines_queue.queue))
                        self.snippet_counter += 1
                        lines_queue.get()

    def _normalize_latex_string(self, string):
        # TODO: actually implement normalization
        return string

    def create_normalized_snippet_from_lines(self, lines):
        snippet_path = os.path.join(self.snippet_dir, "sn_{}".format(self.snippet_counter))
        with open(snippet_path, 'w+') as snippet:
            snippet.write("\\documentstyle[12pt]{article}\n")
            snippet.write("\\begin{document}\n")
            for l in lines:
                snippet.write("{}\n".format(l))
            snippet.write("\\end{document}\n")

    def generate_image_of_snippet(self, snippet_path):


    # 3) method that, given a snippet, returns an image of the rendered pdf
    # 4) method that, given an image, crops it as tightly as possible
    # 5) method that, given a list of images, finds the largest width and height of the images in the set
    # 6) method that, given an image and tuple (width, height), pads the image so that its content is centered and its dimensions are (width, height)

