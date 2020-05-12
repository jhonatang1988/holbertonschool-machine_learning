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
    if type(one_hot) is not np.ndarray:
        return None
    elif one_hot.shape[1] == 0:
        return None
    elif np.amax(one_hot) > 1.0:
        return None

    a_list = []
    for i in range(one_hot.shape[1]):
        max_index = np.argmax(one_hot.T[i])
        a_list.append(max_index)
    return np.array(a_list)
