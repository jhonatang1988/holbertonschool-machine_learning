#!/usr/bin/env python3
"""
updates the weights and biases of a neural network using gradient descent
with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent
    with L2 regularization
    :param Y: one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
    :param weights: is a dictionary of the weights and biases of the neural
    network
    :param cache: is a dictionary of the outputs of each layer of the neural
    network
    :param alpha: is the learning rate
    :param lambtha: L2 regularization parameter
    :param L: is the number of layers of the network
    :return: nothing but the weights and biases of the network should be
    updated in place
    """
    m = Y.shape[1]

    # W1, W2, W3 = weights['W1'], weights['W2'], weights['W3']
    # b1, b2, b3 = weights['b1'], weights['b2'], weights['b3']
    #
    # A0, A1, A2, A3 = cache['A0'], cache['A1'], cache['A2'], cache['A3']
    # # print(A3.shape) (10, 50000)
    # # print(Y.shape) (10, 50000)
    #
    # dZ3 = A3 - Y
    # # print(dZ3)
    #
    # dW3 = (1.0 / m) * np.matmul(dZ3, A2.T) + (lambtha / m) * W3
    # # print(dW3)
    #
    # db3 = (1.0 / m) * np.sum(dZ3, axis=1, keepdims=True)
    # # print(db3)
    #
    # dZ2 = np.matmul(W3.T, dZ3) * (1 - np.power(A2, 2))
    # # print(dZ2)
    #
    # dW2 = (1.0 / m) * np.matmul(dZ2, A1.T) + (lambtha / m) * W2
    # # print(dW2)
    #
    # db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    # # print(db2)
    #
    # dZ1 = np.matmul(W2.T, dZ2) * (1 - np.power(A1, 2))
    # # print(dZ1)
    #
    # dW1 = (1.0 / m) * np.matmul(dZ1, A0.T) + (lambtha / m) * W1
    # # print(dW1)
    #
    # db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    # # print(db1)
    #
    # # update part
    # # print('after updates')
    # W1 -= alpha * dW1
    # W2 -= alpha * dW2
    # W3 -= alpha * dW3
    # b1 -= alpha * db1
    # b2 -= alpha * db2
    # b3 -= alpha * db3

    # now with a loop to make it available to any number of layers
    # first weights
    weights_dict = {}
    for i in range(1, L + 1):
        weights_dict.update({'W' + str(i): weights['W' + str(i)]})
    # print(weights_dict.keys())

    # biases
    biases_dict = {}
    for i in range(1, L + 1):
        biases_dict.update({'b' + str(i): weights['b' + str(i)]})

    # activations and x (A0)
    activations_dict = {}
    for i in range(L + 1):
        activations_dict.update({'A' + str(i): cache['A' + str(i)]})

    # last layer Z derivative
    Z_derivatives = {}
    Z_derivatives.update({'dZ' + str(L): activations_dict['A' + str(L)] - Y})

    # the other Z derivatives
    for i in reversed(range(1, L)):
        # _r is right, ie. next
        W_r = weights_dict['W' + str(i + 1)]
        Z_r = Z_derivatives['dZ' + str(i + 1)]
        A = activations_dict['A' + str(i)]
        Z_derivatives.update(
            {'dZ' + str(i): np.matmul(W_r.T, Z_r) * (1 - np.power(A, 2))})

    # the weights derivatives
    w_derivatives = {}
    for i in reversed(range(1, L + 1)):
        dZ = Z_derivatives['dZ' + str(i)]
        A = activations_dict['A' + str(i - 1)]
        W = weights_dict['W' + str(i)]
        w_derivatives.update(
            {'dW' + str(i): (1.0 / m) * np.matmul(dZ, A.T) + (lambtha / m) * W}
        )

    # biases derivatives
    b_derivatives = {}
    for i in reversed(range(1, L + 1)):
        dZ = Z_derivatives['dZ' + str(i)]
        b_derivatives.update(
            {'db' + str(i): (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)})

    # now the updates loop
    for i in range(1, L + 1):
        weights['W' + str(i)] -= alpha * w_derivatives['dW' + str(i)]
        weights['b' + str(i)] -= alpha * b_derivatives['db' + str(i)]
