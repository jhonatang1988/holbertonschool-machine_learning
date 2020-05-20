#!/usr/bin/env python3
"""
calculates the weighted moving average of a data set
"""

import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    :param data: data set
    :param beta:  weight used for the moving average
    :return: list containing the moving averages of data
    """
    # http: // people.duke.edu / ~ccc14 / sta - 663 - 2018 / notebooks /
    # S09G_Gradient_Descent_Optimization.html
    zs = np.zeros((len(data)))
    z = 0
    for i in range(len(data)):
        z = beta * z + (1 - beta) * data[i]
        zc = z / (1 - beta ** (i + 1))
        zs[i] = zc
    return zs
