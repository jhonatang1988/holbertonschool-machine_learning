#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    :param labels: labels
    :param classes: last dimension of the one-hot matrix must be the number
    of classes
    :return: the one-hot matrix
    """
    return K.utils.to_categorical(y=labels,
                                  num_classes=classes)
