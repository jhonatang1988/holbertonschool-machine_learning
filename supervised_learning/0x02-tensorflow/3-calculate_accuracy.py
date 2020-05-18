#!/usr/bin/env python3
"""
calculates the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    y: param: prediction
    y_pred: param: prediction
    returns: tensor with the accuracy
    """
    # https://stackoverflow.com/questions/42607930/how-to-compute-accuracy-of-cnn-in-tensorflow
    pred = tf.argmax(y_pred, 1)
    eq = tf.equal(pred, tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))

    return acc
