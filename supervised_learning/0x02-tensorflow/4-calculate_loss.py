#!/usr/bin/env python3
"""
calculates the softmax cross-entropy loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    y: param: prediction
    y_pred: param: prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
