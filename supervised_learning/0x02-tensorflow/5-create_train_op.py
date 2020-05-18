#!/usr/bin/env python3
"""
creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    loss: loss of the network
    alpha: learning rate
    return: an operation that trains the network using gradient descent
    """
    # https://stackoverflow.com/questions/45177800/gradient-descent-isnt-working
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return optimizer
