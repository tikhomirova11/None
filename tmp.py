import re
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

tf.keras.backend.clear_session()
K.image_data_format()

def read_image(filename, byteorder='>'):

    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256
                                       else byteorder + 'u2',
                            count=int(width) * int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



img = Image.open(r'C:\data\s1\1.pgm')
img.show()
img = read_image(r'C:\data\s1\1.pgm')
dimensions = img.shape
print('Image Dimension: ', dimensions)

size = 2
total_sample_size = 10000


def get_data(size, total_sample_size):
    # read the image
    image = read_image('data/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    # reduce the size
    image = image[::size, ::size]
    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0

    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(40):
        for j in range(int(total_sample_size / 40)):
            ind1 = 0
            ind2 = 0

            # read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            # read the two images
            img1 = read_image('data/s' + str(i + 1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('data/s' + str(i + 1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            # store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            # as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size / 10)):
        for j in range(10):

            # read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            img1 = read_image('data/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('data/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            # as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1

    # now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y




X, Y = get_data(size, total_sample_size)

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


def build_base_network(input_shape):
    seq = Sequential()

    nb_filter = [6, 12]
    kernel_size = 3

    # convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          padding='valid', data_format='channels_first'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    # convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, padding='valid',data_format='channels_first'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    seq.add(Dropout(.25))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vec_a = base_network(img_a)
feat_vec_b = base_network(img_b)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vec_a, feat_vec_b])

epochs = 13
model = Model([img_a, img_b], distance)

rms = RMSprop()

def contrastive_loss(y_true, y_pred):
    '''Контрастная потеря от Hassell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    '''Вычислить точность классификации с фиксированным порогом по расстояниям.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

img_1 = x_train[:, 0]
img_2 = x_train[:, 1]

model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, epochs=epochs)

