import os

from wand.image import Image, Color

PDF_SNIPPET_DIR = "data/pdf_snippets/"
IMG_SNIPPET_DIR = "data/img_snippets/"

if not os.path.exists(IMG_SNIPPET_DIR):  # create the build directory if not existing
    os.makedirs(IMG_SNIPPET_DIR)


def convert_pdf(filename, output_path, resolution=150):
    """ Convert a PDF into images.

            All the pages will give a single png file with format:
            {pdf_filename}-{page_number}.png

            The function removes the alpha channel from the image and
            replace it with a white background.
        """
    all_pages = Image(filename=filename, resolution=resolution)
    for i, page in enumerate(all_pages.sequence):
        with Image(page) as img:
            img.format = 'png'
            img.background_color = Color('white')
            img.alpha_channel = 'remove'

            image_filename = os.path.splitext(os.path.basename(filename))[0]
            image_filename = '{}-{}.png'.format(image_filename, i)
            image_filename = os.path.join(output_path, image_filename)

            img.save(filename=image_filename)


for subdir, dirs, files in os.walk(PDF_SNIPPET_DIR):
    for f in files:
        f_name = os.path.join(PDF_SNIPPET_DIR, f)
        convert_pdf(f_name, IMG_SNIPPET_DIR, 150)


