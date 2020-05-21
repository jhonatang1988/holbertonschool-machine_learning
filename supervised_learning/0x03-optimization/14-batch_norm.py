#!/usr/bin/env python3
"""
creates a batch normalization layer for a neural network in tensorflow
"""
import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on the output of
    the layer
    :return:
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, activation=activation,
                             name='layer', kernel_initializer=init)

    Z = output(prev)

    Z_mean = np.mean(Z, axis=0)
    Z_variance = np.var(Z, axis=0)
    epsilon = tf.constant(1e-8)
    gamma = tf.Variable(1, shape=[n])
    beta = tf.Variable(0, shape=[n])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=Z_mean, variance=Z_variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)

    return Z_norm
