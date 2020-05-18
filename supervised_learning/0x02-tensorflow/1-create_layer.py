#!/usr/bin/env python3
"""
create layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    create layer
    prev: previous output
    n: number of nodes in each layer
    activation: activation function
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, activation=activation,
                             name='layer', kernel_initializer=init)

    y_pred = output(prev)

    return y_pred
