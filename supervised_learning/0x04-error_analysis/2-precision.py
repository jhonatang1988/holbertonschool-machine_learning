#!/usr/bin/env python3
"""
calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    :param confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: numpy.ndarray of shape (classes,) containing
    the precision of each class
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    return TP / (TP + FP)
