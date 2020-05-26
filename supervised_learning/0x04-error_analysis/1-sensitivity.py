#!/usr/bin/env python3
"""
calculates the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    :param confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: a numpy.ndarray of shape (classes,) containing the
    sensitivity of each class
    """
    # https://stackoverflow.com/questions/43782404
    # /how-to-find-tn-tp-fn-fp-from-matrix-in-python
    # FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    # TN = confusion.sum() - (FP + FN + TP)

    return TP / (TP + FN)
