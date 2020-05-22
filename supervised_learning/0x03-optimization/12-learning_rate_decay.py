#!/usr/bin/env python3
"""
creates a learning rate decay operation in tensorflow using inverse time decay
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow using inverse time
    decay
    :param alpha: learning rate
    :param decay_rate: weight used to determine the rate at which alpha will
    decay
    :param global_step: is the number of passes of gradient descent that have
    elapsed
    :param decay_step:  is the number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       learning_rate=alpha,
                                       staircase=True)
