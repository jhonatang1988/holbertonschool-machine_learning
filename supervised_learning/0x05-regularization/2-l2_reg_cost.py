#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization
    :param cost: cost without L2
    :return: cost with L2 regularization
    """
    reg_losses_list = tf.losses.get_regularization_losses(scope=None)

    return cost + reg_losses_list
