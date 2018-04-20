import os, re

from queue import Queue
from wand.image import Image, Color


class DataGenerator:



    def __init__(self, source, snippet_dest, image_dest, snippet_size, image_resolutation=150):
        self.doc_dir = source
        self.snippet_dir = snippet_dest
        self.image_dir = image_dest
        self.snippet_size = snippet_size
        self.snippet_counter = 0
        self.img_res = image_resolutation

    def generate(self):
        self.snippet_counter = 0
        latex_docs_src = os.fsencode(self.doc_dir)
        # TODO: track the sizes of all the generated cropped images
        for file in os.listdir(latex_docs_src):
            filename = os.fsdecode(file)
            filepath = os.path.join(latex_docs_src, filename)
            # TODO: generates normalized text snippets from well-formed latex document
            # TODO: tightly crop images and track the maximum width and height of images

        # TODO: pad images so that they are uniformly sized

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
                snippet.write("{}\n".format(self._normalize_latex_string(l)))
            snippet.write("\\end{document}\n")

    def _generate_image_from_pdf(self, pdf_path, snippet_name):
        all_pages = Image(filename=pdf_path, resolution=self.img_res)
        image_filename = os.path.join(self.image_dir, snippet_name, '.png')
        for i, page in enumerate(all_pages.sequence):
            with Image(page) as img:
                img.format = 'png'
                img.background_color = Color('white')
                img.alpha_channel = 'remove'
                img.save(filename=image_filename)
        return image_filename

    def generate_image_of_snippet(self, snippet_path):
        snippet_file_name = os.path.basename(snippet_path)
        command = "pdflatex -interaction=batchmode -output-directory={} {}"
        os.system(command.format(self.image_dir, snippet_path))

        pdf_path = os.path.join(self.image_dir, snippet_file_name, ".pdf")
        image_path = self._generate_image_from_pdf(pdf_path, snippet_file_name)

        os.remove(os.path.join(self.image_dir, snippet_file_name, ".aux"))
        os.remove(os.path.join(self.image_dir, snippet_file_name, ".log"))
        os.remove(os.path.join(self.image_dir, snippet_file_name, ".tex"))
        os.remove(pdf_path)
        return image_path


    def full_crop_image(self, image_path):
        # TODO: method that, given an image, crops it as tightly as possible


    def center_pad_image(self, image_path, desired_width, desired_height):
        # TODO: method that, given an image and tuple (width, height), pads the image so that its content is centered and its dimensions are (width, height)


