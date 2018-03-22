import numpy as np
import sys
import os
import math
import codecs
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import LSTM, TimeDistributed, RepeatVector
from keras.utils import np_utils, to_categorical
from keras.callbacks import Callback

np.random.seed(555)
hidden = 100 # hidden layer size
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .")
pairs = list(zip(range(1, len(CHARS)+1), CHARS))
char_to_int = {k:v for (v,k) in pairs}
int_to_char = {k:v for (k,v) in pairs}
int_to_char[0] = '_'

def load_data():
    num_samples = 24389
    train_size = int(0.8*num_samples)
    valid_size = math.ceil(0.1*num_samples)
    test_size = math.ceil(0.1*num_samples)
    data_arr = np.load("post_data.npy")
    target_arr = np.load("post_target.npy")
    data_mat = np.empty((num_samples, 30, len(CHARS)+1))
    target_mat = np.empty((num_samples, 30, len(CHARS)+1))
    for i in range(num_samples):
        data_mat[i] = to_categorical(data_arr[i], len(CHARS) + 1)
        target_mat[i] = to_categorical(target_arr[i], len(CHARS) + 1)
    rand_ind = np.arange(num_samples)
    np.random.shuffle(rand_ind)
    train_ind = rand_ind[:train_size]
    valid_ind = rand_ind[train_size:train_size+valid_size]
    test_ind = rand_ind[-test_size:]
    assert(len(test_ind) == test_size)
    X_train = data_mat[train_ind]
    y_train = target_mat[train_ind]
    X_valid = data_mat[valid_ind]
    y_valid = target_mat[valid_ind]
    X_test = data_mat[test_ind]
    y_test = target_mat[test_ind]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def generate_model(output_len):
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(30, len(CHARS)+1), kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, input_shape=(30, len(CHARS)+1), kernel_initializer="he_normal", return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(output_len))
    model.add(LSTM(hidden, kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden, kernel_initializer="he_normal", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(CHARS)+1, kernel_initializer="he_normal")))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class Metrics(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        precision, recall, f1, _ = precision_recall_fscore_support(np.vstack(val_targ), np.vstack(val_predict))
        print(" — f1: %f — precision: %f — recall %f" %(f1, precision, recall))
        return

metrics = Metrics()

if __name__ == '__main__':
    from_file = None
    if len(sys.argv) > 1:
        from_file = sys.argv[1]
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    model = generate_model(30)
    if from_file:
        model.load_weights(from_file)
        pred = model.predict(X_test)
        predicted = ""
        original = ""
        actual = ""
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                predicted += int_to_char[np.argmax(pred[i][j])]
                original += int_to_char[np.argmax(X_test[i][j])]
                actual += int_to_char[np.argmax(y_test[i][j])]
        print("Original:\n", original)
        print("Predicted:\n", predicted)
        print("Actual:\n", actual)
    else:
        for i in range(75):
            print()
            print('-'*50)
            print('Iteration', i+1)
            model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_valid, y_valid), verbose=1)
    
        score = model.evaluate(X_test, y_test, verbose=1)
        model.save_weights("model_1.h5")
        print(score)
    
