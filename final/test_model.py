import numpy as np
import sys
import os
import math
import codecs
from scipy.sparse import csr_matrix
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Masking
from keras.layers import LSTM, TimeDistributed, RepeatVector
from keras.utils import np_utils, to_categorical
from keras.callbacks import Callback

np.random.seed(555)
hidden = 500 # hidden layer size
MAX_LEN = 50
MIN_LEN = 30
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
CHAR_LEN = 37
pairs = list(zip(range(1, CHAR_LEN+1), CHARS))
char_to_int = {k:v for (v,k) in pairs}
int_to_char = {k:v for (k,v) in pairs}
int_to_char[0] = ""

def load_data(data):
    features = CHAR_LEN + 1
    X_test = np.empty((data.shape[0], MAX_LEN, features))
    data_arr = np.flip(data, 1)
    for i in range(data.shape[0]):
        X_test[i] = to_categorical(data_arr[i], features)
    return X_test

def generate_model(output_len):
    features = CHAR_LEN + 1
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(MAX_LEN, features)))
    model.add(LSTM(hidden, input_shape=(MAX_LEN, features), kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, input_shape=(MAX_LEN, features), kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, input_shape=(MAX_LEN, features), kernel_initializer="he_normal", return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(output_len))
    model.add(LSTM(hidden, kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(features, kernel_initializer="he_normal")))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def test_model(model, X):
    pred = model.predict(X)
    predicted = ""
    original = ""
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            predicted += int_to_char[np.argmax(pred[i][j])]
            original += int_to_char[np.argmax(X[i][j])]
    return predicted

def split_rows(words):
    rows = 1
    row_len = 0
    X = []
    X_mat = []
    for word in words:
        if row_len + len(word) + 1 > MAX_LEN:
            rows += 1
            row_len = 0
            X_mat.append(' '.join(X))
            X = []
        row_len += len(word) + 1
        X.append(word)
    X_mat.append(' '.join(X))
    return X_mat, rows

def predict(input_path, output_path, verbose):
	weights = "model_5.h5"
	file = input_path
	X_text = open(file).read().strip('\n').split()
	X_text, rows = split_rows(X_text)
	X_test = np.empty((rows, MAX_LEN))
	for row in range(rows):
		for i in range(len(X_text[row])):
			X_test[row][i] = (char_to_int[X_text[row][i]])
		for i in range(len(X_text[row]), MAX_LEN):
			X_test[row][i] = 0
	X_test = load_data(X_test)
	model = generate_model(MAX_LEN)
	model.load_weights(weights)
	pred = test_model(model, X_test)
	if verbose:
		print("Predicted:", pred)
	with open(output_path, "w+") as f:
		f.write(pred)

if __name__ == '__main__':
	file = sys.argv[1]
	predict(file, file.split('.')[0]+"_pred.txt")
	print("Done")