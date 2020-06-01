#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    :param labels: labels
    :param classes: last dimension of the one-hot matrix must be the number of classes
    :return: the one-hot matrix
    """
    if not classes:
        return K.utils.to_categorical(labels, classes)
    else:
        return K.utils.to_categorical(labels, len(labels))
