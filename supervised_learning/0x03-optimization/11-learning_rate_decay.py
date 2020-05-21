#!/usr/bin/env python3
"""
updates the learning rate using inverse time decay in numpy
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in numpy
    :param alpha: learning rate
    :param decay_rate: weight used to determine the rate at which alpha will
    decay
    :param global_step: is the number of passes of gradient descent that
    have elapsed
    :param decay_step: the number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the updated value for alpha
    """
    new_alpha = alpha / (1 + decay_rate * np.floor_divide(global_step,
                                                          decay_step))

    return new_alpha
