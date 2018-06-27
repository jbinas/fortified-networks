import keras
import tensorflow as tf
import numpy.random as rng

from keras.datasets import cifar10
from keras.utils import np_utils


def data_cifar10(**kwargs):
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    tpermutation = rng.permutation(X_test.shape[0])

    X_test = X_test[tpermutation]
    y_test = y_test[tpermutation]

    permutation = rng.permutation(X_train.shape[0])

    X_train = X_train[permutation]
    y_train = y_train[permutation]

    X_train /= 255
    X_test /= 255
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def preprocess_image(image, is_training):

    _HEIGHT=32
    _WIDTH=32
    _DEPTH=3

    if is_training:
        """Preprocess a single image of layout [height, width, depth]."""
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    return image
