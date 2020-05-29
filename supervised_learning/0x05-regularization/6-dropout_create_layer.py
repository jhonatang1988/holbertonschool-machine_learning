#!/usr/bin/env python3
"""
creates a layer of a neural network using dropout
"""
import numpy as np
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    :param prev: is a tensor containing the output of the previous layer
    :param n: is the number of nodes the new layer should contain
    :param activation: is the activation function that should be used on the
    layer
    :param keep_prob: is the probability that a node will be kept
    :return: output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop_out = tf.layers.Dropout(rate=keep_prob)
    next_layer = tf.layers.Dense(units=n,
                                 activation=activation,
                                 name='layer',
                                 kernel_initializer=init,
                                 kernel_regularizer=drop_out
                                 )

    y_pred = next_layer(prev)

    return y_pred
