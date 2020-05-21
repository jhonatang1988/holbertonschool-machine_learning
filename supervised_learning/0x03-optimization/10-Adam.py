#!/usr/bin/env python3
"""creates the training operation for a neural network in tensorflow using
the Adam optimization algorithm """

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: small number to avoid ZeroDivisionError
    :return: Adam optimization operation
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)
