#!/usr/bin/env python3
"""creates the training operation for a neural network in tensorflow using
the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates train_op with momentum optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: momentum weight
    :return: momentum optimization operation
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1,
                                      name='train_op').minimize(loss=loss)
