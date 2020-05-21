#!/usr/bin/env python3
"""creates the training operation for a neural network in tensorflow using
the RMSProp optimization algorithm """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: small number to avoid ZeroDivisionError
    :return: RMSProp optimization operation
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha, momentum=beta2,
                                     epsilon=epsilon).minimize(loss)
