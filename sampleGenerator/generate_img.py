import glob
import pickle
from PIL import Image, ImageDraw, ImageFont


TEXT_FILES = "textfile/*"
TTF_FILES = 'ttf/*'
OUTPUT_DIR = "paragraph_fontgen/"


if __name__ == '__main__':
    fontCount = 1
    textCount = 1
    for fontType in glob.glob(TTF_FILES):
        for textfile in glob.glob(TEXT_FILES):
            img = Image.new('L', (1024, 512), color=255)
            fnt = ImageFont.truetype(fontType, 36)
            d = ImageDraw.Draw(img)
            file = open(textfile, "r")
            fileContent = file.read()
            d.text((10, 10), fileContent, font=fnt, fill=0)
            img.save(OUTPUT_DIR + 'font' + str(fontCount) + '_text' + str(textCount) + '.png')
            textCount += 1
        fontCount += 1
