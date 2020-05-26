#!/usr/bin/env python3
"""
creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    :param labels: (m, classes) containing the correct labels for each data
    point
    :param logits: is a one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels
    :return: a confusion numpy.ndarray of shape (classes, classes) with
    row indices representing the correct labels and column indices
    representing the predicted labels
    """
    # https://stackoverflow.com/questions/50021928
    # /build-confusion-matrix-from-two-vector
    classes = np.zeros((labels.shape[1], labels.shape[1]), dtype=float)
    np.add.at(classes, (np.argmax(labels, axis=1), np.argmax(logits,
                                                             axis=1)), 1)

    return classes
