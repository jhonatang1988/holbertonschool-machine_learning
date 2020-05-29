#!/usr/bin/env python3
"""
conducts forward propagation using Dropout
"""
import numpy as np


def _softMax(Z):
    """
    _softMax activation function
    :param Z: Z
    :return: A
    """
    exps = np.exp(Z - np.max(Z))
    return exps / np.sum(exps, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    :param X: shape (nx, m) containing the input data for the network
    :param weights: a dictionary of the weights and biases of the neural
    network
    :param L: number of layers in the network
    :param keep_prob: the probability that a node will be kept
    :return: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer
    """
    # https://wiseodd.github.io/techblog/2016/06/25/dropout/
    # activations
    activations_dict = {}

    # put X as A0
    activations_dict.update({'A0': X})

    # dropouts
    dropout_dict = {}

    # Z (np.matmul(W, A_PREV) + B)
    # A = activation_function(Z)
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        A = activations_dict['A' + str(i - 1)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A) + b
        if i == L:
            activations_dict.update(
                {'A' + str(i): _softMax(Z)})
        else:
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            dropout_dict.update({'D' + str(i): D})
            activations_dict.update(
                {'A' + str(i): np.tanh(Z)})
            A = activations_dict['A' + str(i)]
            A *= D
            A /= keep_prob

        # print(activations_dict['A3'].shape[0])
        # print(activations_dict['A3'].shape[1])

    activations_dict.update(dropout_dict)
    return activations_dict
