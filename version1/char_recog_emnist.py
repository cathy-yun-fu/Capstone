# Mute tensorflow debugging information console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Flatten, Dense
import pickle
import argparse
import keras
import numpy as np
import load_mnist_data
import plot_graph


def build_net(training_data, width=28, height=28, verbose=False):
    ''' Build and train neural network. Also offloads the net in .yaml and the
        weights in .h5 to the bin/.
        Arguments:
            training_data: the packed tuple from load_data()
        Optional Arguments:
            width: specified width
            height: specified height
            epochs: the number of epochs to train over
            verbose: enable verbose printing
    '''
    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32 # number of convolutional filters to use
    pool_size = (2, 2) # size of pooling area for max pooling
    kernel_size = (3, 3) # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if verbose == True: print(model.summary())
    return model


def build_vgg16_model1(training_data, width=28, height=28, verbose=False):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32 # number of convolutional filters to use
    pool_size = (2, 2) # size of pooling area for max pooling
    kernel_size = (3, 3) # convolution kernel size

    model = Sequential()
    model.add(Conv2D(filters=32,
                            kernel_size=kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Conv2D(filters=32,
                            kernel_size=kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu'))
    model.add(Conv2D(filters=64,
                            kernel_size=kernel_size,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(filters=32,
    #                         kernel_size=kernel_size,
    #                         padding='valid',
    #                         activation='relu'))
    # model.add(Conv2D(filters=32,
    #                         kernel_size=kernel_size,
    #                         activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if verbose == True: print(model.summary())
    return model




def build_vgg16_model2(training_data, width=28, height=28):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, pooling='max')
    model_vgg16_conv.summary()

    # Use the generated model
    input_shape = Input(shape=(244, 244, 3), name='image_input')

    output_vgg16_conv = model_vgg16_conv(input_shape)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    model = Model(input=input_shape, output=x)

    # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def train(model, training_data, dir_name, batch_size=256, epochs=30):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    history_callback = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open('bin/' + dir_name + '/model.yaml', "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/' + dir_name + '/model.h5')
    analyze_training(history_callback, dir_name)



def analyze_training(history_callback, dir_name):
    # Callback for analysis in TensorBoard
    output_dir = './Graph/' + dir_name
    keras.callbacks.TensorBoard(log_dir=output_dir, histogram_freq=0, write_graph=True, write_images=True)

    training_loss = np.array(history_callback.history['loss'])
    training_acc = np.array(history_callback.history['acc'])

    validation_loss = np.array(history_callback.history['val_loss'])
    validation_acc = np.array(history_callback.history['val_acc'])

    plot_graph.generate_graph(training_loss, training_acc, validation_loss, validation_acc,
                              title=dir_name, output_dir=output_dir)

def baseline_model(num_pixels):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    # create model
    model = Sequential()
    # densely-connected NN layer.
    num_pixels = 28*28
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(nb_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=200, verbose=2)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data', required=True)
    parser.add_argument('-outdir', type=str, help='Output directory name', required=True)
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin/'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    if not os.path.exists(bin_dir + '/' + args.outdir):
        os.makedirs(bin_dir + '/' + args.outdir)

    training_data = load_mnist_data.load_data(args.file, args.outdir, width=args.width, height=args.height,
                                              max_=args.max, verbose=args.verbose)


    # model = baseline_model(28*28)
    model = build_vgg16_model1(training_data=training_data, width=28, height=28)
    # model = build_net(training_data, width=args.width, height=args.height, verbose=args.verbose)
    train(model, training_data, dir_name=args.outdir, epochs=args.epochs)
