#!/usr/bin/env python3
"""
builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: the number of input features to the network
    :param layers: a list containing the number of nodes in each layer of
    the network
    :param activations: a list containing the activation functions used for
    each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: the probability that a node will be kept for dropout
    :return: keras model
    """

    # init the model
    model = K.Sequential()

    for i in range(layers):
        if i == 0:
            model.add(K.Dense(layers[i]), input_shape=(nx, None),
                      activation=activations[i],
                      kernel_regularizer=K.regularizers.l2)

        model.add(K.Dense(layers[i], activation=activations[i]),
                  kernel_regularizer=K.regularizers.l2)

        if i != layers:
            model.add(K.Dropout(keep_prob))

    return model
