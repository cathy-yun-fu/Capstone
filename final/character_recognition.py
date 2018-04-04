import cv2
import glob
from keras.models import model_from_yaml
import os
from os.path import basename
import pickle
import re
import argparse

# Global variables

output_dir = "CNN/"
MODEL_PATH_BALANCED = 'bin/balanced50_Adadelta_v2/'
MODEL_PATH_LETTER = 'bin/letter50_Adadelta_v2/'
MAPPING_DIST_BALANCED = "bin/balanced_mapping.p"
MAPPING_DIST_LETTER = "bin/letter_mapping.p"
input_dir = "PREPROCESS/"
test_dir = 'test_data/'

def resize_img(im_pth, desired_size):
    im = cv2.imread(im_pth, cv2.COLOR_BGR2GRAY)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = 0
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im


def load_model(model_path):
    # load YAML and create model
    yaml_file = open(model_path + 'model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(model_path + "model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return loaded_model


def alphanumeric_sort(list):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list, key=alphanum_key)

def char_recognition(args, input_dir, output_dir):
    # Define path
    if args.test:
        img_dir_path = test_dir
    else:
        img_dir_path = input_dir

    # Create output directory if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.balanced:
        model_path = MODEL_PATH_BALANCED
    else:
        model_path = MODEL_PATH_LETTER

    model = load_model(model_path)
    text_name = ''
    for para_path in alphanumeric_sort(glob.glob(img_dir_path)):
        if ".png" not in para_path:
            sentence = []
            text_name = str(basename(para_path).split(';')[0])
            for word_im_pth in alphanumeric_sort(glob.glob(para_path + '/Row*/Word*/')):
                for im_pth in alphanumeric_sort(glob.glob(word_im_pth + '*.jpg')):
                    new_img = resize_img(im_pth, 28)
                    new_img = new_img / 255
                    new_img = new_img.reshape(1, 28, 28, 1)
                    if args.balanced:
                        label = model.predict_classes(new_img)[0] + 1
                        mapping = pickle.load(open(MAPPING_DIST_BALANCED, "rb"))
                    else:
                        label = model.predict_classes(new_img)[0] + 1
                        mapping = pickle.load(open(MAPPING_DIST_LETTER, "rb"))
                    character = chr(mapping[label])
                    sentence.append(character.lower())
                sentence.append(' ')

            with open(output_dir + basename(para_path) + '.txt', "w") as text_file:
                text_file.write(''.join(sentence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A script for predicting the image with model.yaml')
    parser.add_argument('--test', action='store_true', default=False, help='use test folder')
    parser.add_argument('--letter', action='store_true', default=False, help='use letter model')
    parser.add_argument('--balanced', action='store_true', default=False, help='use balanced model with numbers')
    args = parser.parse_args()

    char_recognition(args, input_dir, output_dir)