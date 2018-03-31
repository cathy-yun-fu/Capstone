import cv2
import glob
from keras.models import model_from_yaml
import os
from os.path import basename
import pickle
import re
import argparse


# Global variables
DESIRED_SIZE = 28
MAPPING_DIST = "bin/balanced_mapping.p"
OUTPUT_DIR = "predicted_paragraph/"
MODEL_PATH = 'bin/old/balanced30_v2/'
PARAGRAPH_DIR = '../sampleGenerator/ROOT_DIR/*'
TEST_DIR = 'test_data/'
TARGET_FILE_PATH = '../sampleGenerator/textfile/'

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

    cv2.imwrite('resized.png', new_im)
    # cv2.imshow('resized', new_im)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
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


def get_accuracy(predicted, target_path):
    file = open(TARGET_FILE_PATH + target_path + '.txt', "r")
    fileContent = file.read()
    target = list(fileContent)

    if len(target) > len(predicted):
        arr = predicted
    else:
        arr = target

    count = 0
    for index in range(len(arr)):
        if predicted[index] == target[index]:
            count += 1
    acc = str(count/len(arr))
    print("accuracy is: " + acc)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A script for predcting the image with model.yaml')
    parser.add_argument('--test', action='store_true', default=False, help='use test folder')
    args = parser.parse_args()

    # Define path
    if args.test:
        img_dir_path = TEST_DIR
    else:
        img_dir_path = PARAGRAPH_DIR

    # Create output directory if not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Remove acc.txt if exist
    try:
        os.remove(OUTPUT_DIR + 'acc.txt')
    except OSError:
        pass

    model = load_model(MODEL_PATH)
    text_name = ''
    for para_path in alphanumeric_sort(glob.glob(img_dir_path)):
        if ".png" not in para_path:
            sentence = []
            text_name = str(basename(para_path).split(';')[0])
            for word_im_pth in alphanumeric_sort(glob.glob(para_path + '/Row*/Word*/')):
                for im_pth in alphanumeric_sort(glob.glob(word_im_pth + '*.jpg')):
                    print('predicting image ', im_pth)
                    new_img = resize_img(im_pth, DESIRED_SIZE)
                    new_img = new_img / 255
                    new_img = new_img.reshape(1, 28, 28, 1)
                    label = model.predict_classes(new_img)
                    mapping = pickle.load(open(MAPPING_DIST, "rb"))
                    character = chr(mapping[label[0]])
                    sentence.append(character.lower())
                    print('predicted char:', character)
                sentence.append(' ')
            acc_msg = "accuracy for " + text_name + " is: " + get_accuracy(sentence, text_name) + '\n'
            with open(OUTPUT_DIR + 'acc.txt', "a") as text_file:
                text_file.write(acc_msg)
            with open(OUTPUT_DIR + text_name + '.txt', "w") as text_file:
                text_file.write(''.join(sentence))
