#!/usr/bin/env python3
"""
updates the weights of a neural network with Dropout regularization using
gradient descent
"""
import numpy as np


def _tanh_derivative(A):
    """
    derivative of tanh function
    :param A: tanh (A)
    :return: derivative of tanh
    """
    return 1 - A ** 2


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization using
    gradient descent
    :param Y: is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
    :param weights: is a dictionary of the weights and biases of the
    neural network
    :param cache:  is a dictionary of the outputs and dropout masks
    of each layer of the neural network
    :param alpha: the learning rate
    :param keep_prob: the probability that a node will be kept
    :param L: number of layers of the network
    :return: nothing but updates weights
    """
    # https://wiseodd.github.io/techblog/2016/06/25/dropout/
    # temporary weights
    weights_delta = {}

    # last layer first - no dropout
    A = cache['A' + str(L)]
    A_left = cache['A' + str(L - 1)]
    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    dZ = A - Y
    dW = 1 / len(Y[0]) * (np.matmul(A_left, dZ.transpose()))
    db = 1 / len(Y[0]) * np.sum(dZ, axis=1, keepdims=True)

    weights_delta.update({'W' + str(L): W - (alpha * dW.T)})
    weights_delta.update({'b' + str(L): b - (alpha * db)})

    # back propagation for hidden layers - yes dropout
    for i in reversed(range(1, L)):
        A = cache['A' + str(i)]
        D = cache['D' + str(i)]
        dZ_right = dZ
        W_right = W
        A_left = cache['A' + str(i - 1)]
        dg = _tanh_derivative(A)
        dZ = (np.matmul(W_right.T, dZ_right)) * dg
        dZ *= D
        dZ /= keep_prob
        dW = 1 / len(Y[0]) * (np.matmul(dZ, A_left.transpose()))
        db = 1 / len(Y[0]) * (np.sum(dZ, axis=1, keepdims=True))

        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        weights_delta.update({'W' + str(i): W - (alpha * dW)})
        weights_delta.update({'b' + str(i): b - (alpha * db)})

    weights.update(weights_delta)
