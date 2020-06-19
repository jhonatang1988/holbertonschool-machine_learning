#!/usr/bin/env python3
"""
builds a dense block as described in Densely Connected Convolutional Networks
https://arxiv.org/pdf/1608.06993.pdf
DenseNet-B implementation
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected Convolutional Networks
    :param X: the output from the previous layer
    :param nb_filters: is an integer representing the number of filters in X
    :param growth_rate: is the growth rate for the dense block
    :param layers: is the number of layers in the dense block
    :return: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """

    init = K.initializers.he_normal()

    for layer_index in range(layers):
        batch_normalization_ = K.layers.BatchNormalization(
            axis=-1
        )

        batch_normalization = batch_normalization_(X)

        activation_ = K.layers.Activation(
            activation='relu'
        )

        activation = activation_(batch_normalization)

        conv2d_ = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=[1, 1],
            kernel_initializer=init,
            padding='same',
        )

        conv2d = conv2d_(activation)

        batch_normalization_1_ = K.layers.BatchNormalization(
            axis=-1
        )

        batch_normalization_1 = batch_normalization_1_(conv2d)

        activation_1_ = K.layers.Activation(
            activation='relu'
        )

        activation_1 = activation_1_(batch_normalization_1)

        conv2d_1_ = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=[3, 3],
            kernel_initializer=init,
            padding='same',
        )

        conv2d = conv2d_1_(activation_1)

        concatenate = K.layers.concatenate([X, conv2d])

        X = concatenate
        nb_filters += growth_rate

    return concatenate, nb_filters
