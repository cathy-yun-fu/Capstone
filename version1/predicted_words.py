from PIL import Image, ImageOps
import cv2
import glob
from keras.models import Sequential, load_model, model_from_yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Merge, LSTM
import yaml
import pickle

# Global variables
DESIRED_SIZE = 28
MAPPING_DIST = "bin/balanced_mapping.p"
OUTPUT_FILE = "Output.txt"
MODEL_PATH = 'bin/old/balanced30_v2/'

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
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    sentence = []
    for word_im_pth in sorted(glob.glob('../ROOT_DIR/paragraph1/Row*/Word*/')):
        for im_pth in sorted(glob.glob(word_im_pth + '*.jpg')):
            print('predicting image ', im_pth)
            new_img = resize_img(im_pth, DESIRED_SIZE)
            new_img = new_img / 255
            new_img = new_img.reshape(1, 28, 28, 1)
            label = model.predict_classes(new_img)
            mapping = pickle.load(open(MAPPING_DIST, "rb"))
            character = chr(mapping[label[0]])
            sentence.append(character)
            print('predicted char:', character)
        sentence.append(' ')

    print(sentence)
    with open(OUTPUT_FILE, "w") as text_file:
        text_file.write(''.join(sentence))
