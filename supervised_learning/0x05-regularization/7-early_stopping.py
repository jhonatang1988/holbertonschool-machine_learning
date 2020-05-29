#!/usr/bin/env python3
"""
determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    :param cost: is the current validation cost of the neural network
    :param opt_cost: is the lowest recorded validation cost of the
    neural network
    :param threshold: is the threshold used for early stopping
    :param patience: is the patience count used for early stopping
    :param count: is the count of how long the threshold has not been met
    :return:  a boolean of whether the network should be stopped early,
    followed by the updated count
    """
    count += 1
    if opt_cost - cost <= threshold:
        if patience > count:
            return False, count
        if patience == count:
            return True, count
        else:
            return False, 0
    else:
        return False, 0
