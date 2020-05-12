#!/usr/bin/env python3
"""
converts a one-hot matrix into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    :param one_hot: one hot encoded matrix
    :return: decoded one-hot matrix
    """
    # if type(one_hot) is not np.ndarray:
    #     return None
    # elif one_hot.shape[1] == 0:
    #     return None
    # elif len(one_hot.shape) != 2:
    #     return None
    # elif np.amax(one_hot) > 1.0:
    #     return None
    # elif np.sum(one_hot) > one_hot.shape[1]:
    #     return None
    # for class_vector in one_hot.T:
    #     # print(np.sum(class_vector))
    #     if np.sum(class_vector) != 1.0:
    #         return None
    if type(one_hot) is not np.ndarray or len(one_hot) == 0:
        return None

        # has to be shape 2 => (classes, m)
    if len(one_hot.shape) != 2:
        return None

        # only can contain 1 and  0
    b = np.where((one_hot != 0) & (one_hot != 1), True, False)
    if b.any() is True:
        return None

    # only can has one 1 per column
    b = one_hot.T.sum(axis=1)
    b = np.where(b > 1, True, False)
    if b.any() is True:
        return None

    a_list = []
    for i in range(one_hot.shape[1]):
        max_index = np.argmax(one_hot.T[i])
        a_list.append(max_index)
    one_hot_dec = np.asarray(a_list)
    return one_hot_dec
