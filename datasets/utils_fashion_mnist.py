
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings
#import cPickle as pickle
import pickle
import gzip
from keras.utils.np_utils import to_categorical   


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def fashion_mnist(datadir='/data/milatmp1/lambalex/fashion_mnist/', train_start=0, train_end=60000, test_start=0,
               test_end=10000):
    """
    Load and preprocess MNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    X_train, Y_train = load_mnist(datadir,kind='train')
    X_test, Y_test = load_mnist(datadir,kind='t10k')

    X_train = X_train.reshape((-1,28,28,1)).astype('float32')/255.0
    X_test = X_test.reshape((-1,28,28,1)).astype('float32')/255.0

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    print(X_train.min(),X_train.max())

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":

    fashion_mnist()

