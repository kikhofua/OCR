import os, re, cv2
import numpy as np

from queue import Queue
from pdf2image import convert_from_path

from data.generation.utils import \
    bad_latex_tokens_regex, begin_block, begin_latex_document, end_latex_document, \
    entire_block, math_block_regex, inline_math_regex, valid_math_token


# TODO: need to make sure to wrap ^ and _ in {}s

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
        for latex_doc in os.listdir(self.doc_dir):
            doc_name = os.fsdecode(latex_doc)
            doc_path = os.path.join(self.doc_dir, doc_name)
            self.extract_snippets_from_latex_document(doc_path)

        max_rows = max_cols = 0
        largest_photo = None
        for snip in os.listdir(self.snippet_dir):
            snip_name = os.fsdecode(snip)
            snip_path = os.path.join(self.snippet_dir, snip_name)
            image_path = self.generate_image_of_snippet(snip_path)
            if os.path.exists(image_path):
                snip_num_rows, snip_num_cols = self.tightly_crop_image(image_path)
                max_rows = max(max_rows, snip_num_rows)
                max_cols = max(max_cols, snip_num_cols)
                largest_photo = image_path if snip_num_rows * snip_num_cols > max_rows * max_cols else largest_photo
            # break
        print(largest_photo)

        for image in os.listdir(self.image_dir):
            image_name = os.fsdecode(image)
            image_path = os.path.join(self.image_dir, image_name)
            self.pad_image_for_consistent_top_left_start(image_path, max_rows, max_cols)

    def _sparsify_inline_maths(self, line):
        dollar_sign_indices = [0] + [i for i, c in enumerate(line) if c == "$"] + [len(line)]
        sparsified_line = line[:dollar_sign_indices[1]]
        for i in range(1, len(dollar_sign_indices) - 2):
            start = dollar_sign_indices[i]
            end = dollar_sign_indices[i+1]
            if i % 2 == 1:
                sparsified_line += self._sparsify_math_blocks(line[start:end+1])
            else:
                sparsified_line += line[start+1: end]
        sparsified_line += line[dollar_sign_indices[-2]+1:]
        return sparsified_line

    def _sparsify_math_blocks(self, math_text):
        tokens = re.findall(valid_math_token, math_text)
        return " ".join(tokens)

    def create_snippet_from_lines(self, lines):
        '''
        Creates a well-formed snippet from @lines.
        Snippets are normalized via @self._remove_bad_tokens method and are saved to
        @self.snippet_dir.

        :param lines: a list of strings that represents the content of the snippet
        :return:
        '''
        snippet_path = os.path.join(self.snippet_dir, "sn_{}".format(self.snippet_counter))
        with open(snippet_path, 'w+') as snippet:
            snippet.write("\\documentstyle[12pt]{article}\n")
            snippet.write("\\usepackage{amsmath, amsthm, amssymb}\n")
            snippet.write("\\pagestyle{empty}\n")
            snippet.write("\\begin{document}\n")
            concat_cont = " ".join(lines)
            snippet.write("{}\n".format(concat_cont))
            snippet.write("\\end{document}\n")

    def extract_snippets_from_latex_document(self, filepath):
        '''
        Creates snippets from the latex documents in @self.doc_dir by sliding a window
        through the lines of the @filepath document of @size self.snippet_size.
        Snippets are normalized via @self._remove_bad_tokens method and are saved to
        @self.snippet_dir.

        :param filepath: path to well formed latex document
        :return:
        '''

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
                    line = re.sub(bad_latex_tokens_regex, "", line)
                    if not line or line.startswith("%"):
                        continue
                    if end_latex_document in line:
                        break
                    if not building_block:
                        matches_begin_block = re.search(begin_block, line)
                        matches_inline_math = re.search(inline_math_regex, line)
                        if matches_begin_block:
                            building_block = True
                        elif matches_inline_math:
                            line = self._sparsify_inline_maths(line)
                        elif "\\" not in line or "$" not in line:  # skip the line if there's nothing interesting
                            continue
                    if building_block:
                        new_block = multi_line_builder + line
                        match = re.search(entire_block, new_block)
                        if match:
                            multi_line_builder = ""
                            building_block = False
                            math_block = re.search(math_block_regex, new_block)
                            if math_block:
                                line = self._sparsify_math_blocks(new_block)
                            else:
                                line = new_block
                        else:
                            multi_line_builder += "{}\n".format(line)
                            continue
                    lines_queue.put(line)
                    if lines_queue.full():
                        self.create_snippet_from_lines(list(lines_queue.queue))
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
        pages = convert_from_path(pdf_path, self.img_res)
        image_filename = os.path.join(self.image_dir, snippet_name + '.jpg')
        for page in pages:
            page.save(image_filename, 'JPEG')
        return image_filename

    def generate_image_of_snippet(self, snippet_path):
        '''
        Generates a PNG image of the PDF that is produced by compiling the
        Latex snippet at @snippet_path.
        The PNG has the same file name as @snippet_path and is written to
        @self.snippet_dir.

        :param snippet_path:
        :return:
        '''
        snippet_file_name = os.path.basename(snippet_path)
        command = "pdflatex -interaction=batchmode -output-directory={} {}"
        pdf_path = os.path.join(self.image_dir, snippet_file_name + ".pdf")
        print("Generating: {}😎".format(pdf_path))
        os.system(command.format(self.image_dir, snippet_path))
        image_path = self._generate_image_from_pdf(pdf_path, snippet_file_name)

        # we either want the snippet and the image or neither
        if os.path.exists(image_path):
            os.remove(pdf_path)
        else:
            os.remove(snippet_path)

        aux_path = os.path.join(self.image_dir, snippet_file_name + ".aux")
        if os.path.exists(aux_path):
            os.remove(aux_path)

        log_path = os.path.join(self.image_dir, snippet_file_name + ".log")
        if os.path.exists(log_path):
            os.remove(log_path)

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
        num_cols = right_bound - left_bound
        num_rows = bottom_bound - top_bound
        return num_rows, num_cols

    def pad_image_for_consistent_top_left_start(self, image_path, desired_rows, desired_cols):
        '''
        Inserts the content of the image at @image_path on a white background
        of size @desired_height x @desired_width.
        This new image then gets further padded with a margin of size @margin
        so that the content does not directly tough the image border.
        This image overwrites the existing image at @image_path

        :param image_path: the path of the file to be padded
        :param desired_rows: the desired width the new image BEFORE the margin is applied
        :param desired_cols: the desired height the new image BEFORE the margin is applied
        :param padding: the amount of extra padding the image should have
        :return:
        '''
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_rows, img_cols = gray_img.shape
        padded_image = np.ones((desired_rows + 2 * self.img_padding, desired_cols + 2 * self.img_padding)) * 255
        padded_image[self.img_padding:img_rows+self.img_padding, self.img_padding:img_cols+self.img_padding] = gray_img
        cv2.imwrite(image_path, padded_image)


source = "data\\example\\"
snippet_destination = "data\\snippets\\"
image_destination = "data\\images\\"
snippet_size = 4
padding = 10
resolution = 150

dg = DataGenerator(source, snippet_destination, image_destination, snippet_size, resolution, padding)
dg.generate()