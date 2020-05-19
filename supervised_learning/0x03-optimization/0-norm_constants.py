#!/usr/bin/env python3
"""
normalizes (standardizes) a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    normalizes (standardizes) a matrix
    :param X: shape (m, nx) to normalize
    :return: mean and standard deviation of the matrix
    """
    mean2d = np.mean(X, axis=0, keepdims=True)
    mean = mean2d.flatten()
    std2d = np.std(X, axis=0, keepdims=True)
    std = std2d.flatten()

    return mean, std
