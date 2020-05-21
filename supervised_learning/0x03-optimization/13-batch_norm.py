#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using batch normalization
    :param Z: numpy.ndarray of shape (m, n) that should be normalized (m is
    the number of data points, n is the number of features in Z
    :param gamma: numpy.ndarray of shape (1, n) containing the scales used for
    batch normalization
    :param beta: numpy.ndarray of shape (1, n) containing the offsets used for
    batch normalization
    :param epsilon: small number used to avoid division by zero
    :return: normalized Z matrix
    """
    # https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    Z_mean = np.mean(Z, axis=0)
    Z_variance = np.var(Z, axis=0)

    Z_norm = (Z - Z_mean) / np.sqrt(Z_variance + epsilon)
    out = gamma * Z_norm + beta

    return out
