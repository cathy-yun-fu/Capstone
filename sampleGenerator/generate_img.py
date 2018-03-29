import glob
import os
from os.path import basename
from PIL import Image, ImageDraw, ImageFont


TEXT_FILES = "textfile/*"
TTF_FILES = 'ttf/Jenny_font2.ttf'
OUTPUT_DIR = "paragraph_fontgen/"


if __name__ == '__main__':
    fontCount = 1
    for fontType in glob.glob(TTF_FILES):
        for textfile in glob.glob(TEXT_FILES):
            text_name = basename(os.path.splitext(textfile)[0])
            img = Image.new('L', (1024, 512), color=255)
            fnt = ImageFont.truetype(fontType, 36)
            draw = ImageDraw.Draw(img)
            file = open(textfile, "r")
            fileContent = file.read()
            draw.text((10, 10), fileContent, font=fnt, fill=0)
            img.save(OUTPUT_DIR + text_name + ';font' + str(fontCount) + '.png')
        fontCount += 1
