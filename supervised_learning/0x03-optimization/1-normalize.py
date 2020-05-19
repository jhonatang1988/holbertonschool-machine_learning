#!/usr/bin/env python3
"""
normalize (standardizes) a matrix
"""
import numpy as np


def normalize(X, m, s):
    """
    normalize (standardizes) a matrix
    :param X: (d, nx) data to normalize
    :param m: mean
    :param s: stddev
    :return: the normalized matrix
    """
    Norm = (X - m) / s
    return Norm
