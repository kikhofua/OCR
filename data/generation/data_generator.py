import os, re, cv2
import numpy as np

from queue import Queue
from wand.image import Image, Color


class DataGenerator:
    def __init__(self, source, snippet_dest, image_dest, snippet_size, image_resolutation=150, image_padding=10):
        self.doc_dir = source
        self.snippet_dir = snippet_dest
        self.image_dir = image_dest
        self.snippet_size = snippet_size
        self.snippet_counter = 0
        self.img_res = image_resolutation
        self.img_padding = image_padding

    def generate(self):
        self.snippet_counter = 0
        latex_docs_src = os.fsencode(self.doc_dir)
        for latex_doc in os.listdir(latex_docs_src):
            doc_name = os.fsdecode(latex_doc)
            doc_path = os.path.join(latex_docs_src, doc_name)
            self.extract_snippets_from_latex_document(doc_path)

        max_width = max_height = 0
        snippets_src = os.fsencode(self.snippet_dir)
        for snip in os.listdir(snippets_src):
            snip_name = os.fsdecode(snip)
            snip_path = os.path.join(snippets_src, snip_name)
            image_path = self.generate_image_of_snippet(snip_path)
            snip_width, snip_height = self.tightly_crop_image(image_path)
            max_width = max(max_width, snip_width)
            max_height = max(max_height, snip_height)

        image_src = os.fsencode(self.image_dir)
        for image in os.listdir(image_src):
            image_name = os.fsdecode(image)
            image_path = os.path.join(image_src, image_name)
            self.pad_image_for_consistent_top_left_start(image_path, max_width, max_height)

    def _normalize_latex_string(self, string):
        # TODO: actually implement normalization
        return string

    def create_normalized_snippet_from_lines(self, lines):
        '''
        Creates a well-formed snippet from @lines.
        Snippets are normalized via @self._normalize_latex_string method and are saved to
        @self.snippet_dir.

        :param lines: a list of strings that represents the content of the snippet
        :return:
        '''
        snippet_path = os.path.join(self.snippet_dir, "sn_{}".format(self.snippet_counter))
        with open(snippet_path, 'w+') as snippet:
            snippet.write("\\documentstyle[12pt]{article}\n")
            # TODO: include "\usepackage{amsmath}
            snippet.write("\\begin{document}\n")
            for l in lines:
                snippet.write("{}\n".format(self._normalize_latex_string(l)))
            snippet.write("\\end{document}\n")

    def extract_snippets_from_latex_document(self, filepath):
        '''
        Creates snippets from the latex documents in @self.doc_dir by sliding a window
        through the lines of the @filepath document of @size self.snippet_size.
        Snippets are normalized via @self._normalize_latex_string method and are saved to
        @self.snippet_dir.

        :param filepath: path to well formed latex document
        :return:
        '''
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

    def _generate_image_from_pdf(self, pdf_path, snippet_name):
        '''
        Creates a PNG from the PDF located at @pdf_path and saves it at
        to self.image_dir

        :param pdf_path: path of the PDF to make a PNG of
        :param snippet_name: name of the snippet latex document (PNG will have the same name)
        :return:
        '''
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

    @staticmethod
    def tightly_crop_image(image_path):
        '''
        Crops the image at @image_path (overwrites the file) so that it tightly bounds
        the latex content.

        :param image_path: path of the image to be cropped
        :return: the width and height of the post-cropped image
        '''
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        pixel_is_text = gray_img < 255
        text_rows, text_cols = np.nonzero(pixel_is_text)
        left_bound, right_bound = min(text_cols), max(text_cols)
        top_bound, bottom_bound = min(text_rows), max(text_rows)
        tightly_cropped = gray_img[top_bound:bottom_bound, left_bound:right_bound]
        cv2.imwrite(image_path, tightly_cropped)
        width = right_bound - left_bound
        height = bottom_bound - top_bound
        return width, height

    def pad_image_for_consistent_top_left_start(self, image_path, desired_width, desired_height):
        '''
        Inserts the content of the image at @image_path on a white background
        of size @desired_height x @desired_width.
        This new image then gets further padded with a margin of size @margin
        so that the content does not directly tough the image border.
        This image overwrites the existing image at @image_path

        :param image_path: the path of the file to be padded
        :param desired_width: the desired width the new image BEFORE the margin is applied
        :param desired_height: the desired height the new image BEFORE the margin is applied
        :param padding: the amount of extra padding the image should have
        :return:
        '''
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_width, img_height = gray_img.shape
        padded_image = np.ones((desired_width + 2*self.img_padding, desired_height + 2*self.img_padding)) * 255
        padded_image[self.img_padding:img_width+self.img_padding, self.img_padding:img_height+self.img_padding] = gray_img
        cv2.imwrite(image_path, padded_image)




