#!/usr/bin/env python3
"""
builds a neural network with the Keras library
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: the number of input features to the network
    :param layers: a list containing the number of nodes in each layer of the
    network
    :param activations: a list containing the activation functions used for
    each layer of the network
    :param lambtha: the L2 regularization parameter
    :param keep_prob: the probability that a node will be kept for dropout
    :return: keras model
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    x = inputs

    for i in range(len(layers)):

        x = K.layers.Dense(units=layers[i],
                           activation=activations[i],
                           kernel_regularizer=reg)(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
