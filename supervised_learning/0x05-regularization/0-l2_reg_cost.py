#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    :param cost: cost of the network without L2 regularization
    :param lambtha: the regularization parameter
    :param weights: is a dictionary of the weights and biases (
    numpy.ndarrays) of the neural network
    :param L: the number of layers in the neural network
    :param m: the number of data points used
    :return: the cost of the network accounting for L2 regularization
    """
    # print(cost)
    # print(lambtha)
    # print(weights)
    # print(L)
    # print(m)
    # W1 = np.sum(np.square(weights['W1']) ** 2)
    # print(W1)
    W1_L2 = np.linalg.norm(weights['W1'], ord='fro')
    # print(W1_L2)
    W2_L2 = np.linalg.norm(weights['W2'], ord='fro')
    W3_L2 = np.linalg.norm(weights['W3'], ord='fro')
    L2_regularization_cost = (W1_L2 + W2_L2 + W3_L2) * (lambtha / (2 * m))

    return cost + L2_regularization_cost
