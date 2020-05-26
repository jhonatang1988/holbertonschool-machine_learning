#!/usr/bin/env python3
"""
calculates the F1 score of a confusion matrix
"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
    :param confusion: numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the
    predicted labels
    :return: numpy.ndarray of shape (classes,) containing the F1
    score of each class
    """
    # https://en.wikipedia.org/wiki/F1_score
    return 2 * ((precision(confusion) * sensitivity(confusion)) / (
            precision(confusion) + sensitivity(confusion)))
