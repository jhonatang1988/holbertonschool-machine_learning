#!/usr/bin/env python3
"""
creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on the output
    of the layer
    :return:
    """
    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch
    # -normalization-in-tensorflow
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n,
                             name='layer', kernel_initializer=init)

    Z = output(prev)
    mean, variance = tf.nn.moments(Z, [0])

    epsilon = 1e-8

    beta = tf.Variable(tf.constant(0.0, shape=[n]))
    gamma = tf.Variable(tf.constant(1.0, shape=[n]))
    Z_norm = tf.nn.batch_normalization(
        x=Z, mean=mean, variance=variance, offset=beta, scale=gamma,
        variance_epsilon=epsilon)

    if not activation:
        return Z_norm
    else:
        return activation(Z_norm)
