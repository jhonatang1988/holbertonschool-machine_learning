#!/usr/bin/env python3
"""
converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    :param Y:   classes
    :param classes: max number of classes
    :return: a one-hot encoding of Y shape (classes, m) or NONE if failure
    """
    a_list = []
    for i in range(classes):
        array = np.zeros(classes)
        # ahora incluimos en el indice
        index = Y[i]
        array[index] = 1
        a_list.append(array)
    one_hot_encode = np.array(a_list).T
    return one_hot_encode
