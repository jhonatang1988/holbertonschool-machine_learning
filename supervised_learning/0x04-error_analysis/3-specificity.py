#!/usr/bin/env python3
"""
calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    :param confusion: is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: numpy.ndarray of shape (classes,) containing the specificity
    of each class
    """
    # Specificity or true negative rate
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return TN / (TN + FP)
