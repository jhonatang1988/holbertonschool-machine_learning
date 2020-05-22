#!/usr/bin/env python3
"""
updates a variable using the gradient descent with momentum optimization
algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum optimization
    algorithm
    :param alpha: the learning rate
    :param beta1: the momentum weight
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: is the previous first moment of var
    :return: updated variable and the new moment, respectively
    """
    # https://hackernoon.com/implementing-different-variants-of-gradient-
    # descent-optimization-algorithm-in-python-using-numpy-809e7ab3bab4
    # http://people.duke.edu/~ccc14/sta-663-2018/notebooks/
    # S09G_Gradient_Descent_Optimization.html

    velocity = beta1 * v + (1 - beta1) * grad
    new_var = var - alpha * velocity

    return new_var, velocity
