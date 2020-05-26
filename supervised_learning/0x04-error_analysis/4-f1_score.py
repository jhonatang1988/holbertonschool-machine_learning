#!/usr/bin/env python3
"""
calculates the F1 score of a confusion matrix
"""
import numpy as np


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
    :param confusion: numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the
    predicted labels
    :return: numpy.ndarray of shape (classes,) containing the F1
    score of each class
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    TP = np.diag(confusion)
    PPV = TP / (TP + FP)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TPR = TP / (TP + FN)

    return 2 * ((PPV * TPR) / (PPV + TPR))
