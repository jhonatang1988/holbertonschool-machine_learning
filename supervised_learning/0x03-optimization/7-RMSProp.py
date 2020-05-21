#!/usr/bin/env python3
"""
updates a variable using the RMSProp optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: small number to avoid ZeroDivisionError
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param s: is the previous second moment of var
    :return: the updated variable and the new moment, respectively
    """
    # http://people.duke.edu/~ccc14/sta-663-2018/notebooks/
    # S09G_Gradient_Descent_Optimization.html
    velocity = beta2 * s + (1 - beta2) * grad ** 2
    newMoment = var - alpha * grad / (epsilon + np.sqrt(velocity))

    return newMoment, velocity
