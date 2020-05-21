#!/usr/bin/env python3
"""
updates a variable in place using the Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm
    :param alpha: learning rate
    :param beta1: the weight used for the first moment
    :param beta2: the weight used for the second moment
    :param epsilon: is a small number to avoid ZeroDivisionError
    :param var: is a numpy.ndarray containing the variable to be updated
    :param grad: is a numpy.ndarray containing the gradient of var
    :param v: is the previous first moment of var
    :param s: is the previous second moment of var
    :param t: is the time step used for bias correction
    :return: the updated variable, the new first moment, and the new second
    moment, respectively
    """
    # https://hackernoon.com/implementing-different-variants-of-
    # gradient-descent-optimization-algorithm-in-python-using-numpy-809e7ab3bab4
    momentum_v = beta1 * v + ((1 - beta1) * grad)
    decay_v = beta2 * s + ((1 - beta2) * grad ** 2)

    new_momentum_v = momentum_v / (1 - np.power(beta1, t))
    new_decay_v = decay_v / (1 - np.power(beta2, t))

    var -= (alpha / (np.sqrt(new_decay_v) + epsilon)) * new_momentum_v

    return var, momentum_v, decay_v
