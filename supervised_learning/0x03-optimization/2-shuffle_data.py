#!/usr/bin/env python3
"""
shuffles the data points in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    :param X: first matrix
    :param Y: second matrix
    :return: both matrices shuffled
    """
    # https://stackoverflow.com/questions/43229034/randomly-shuffle-data-
    # and-labels-from-different-files-in-the-same-order
    idx = np.random.permutation(Y.shape[0])
    return X[idx], Y[idx]
