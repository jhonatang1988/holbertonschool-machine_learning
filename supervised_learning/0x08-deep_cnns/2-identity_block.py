#!/usr/bin/env python3
"""
builds an identity block as described in Deep Residual Learning for Image
Recognition (2015)
https://arxiv.org/pdf/1512.03385.pdf
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in Deep Residual Learning for Image
    :param A_prev: the output from the previous layer
    :param filters: a tuple or list containing F11, F3, F12, respectively:
    F11 is the number of filters in the first 1x1 convolution
    F3 is the number of filters in the 3x3 convolution
    F12 is the number of filters in the second 1x1 convolution
    :return: the activated output of the identity block
    """
    # left side of the image
    # https://www.dropbox.com/s/ydsblgks4ce2073/
    # Identity%20Block%20resnet.png?dl=0
    F11, F3, F12 = filters

    init = K.initializers.he_normal()

    # conv2d 1x1 + 1(S)
    conv2d_ = K.layers.Conv2D(
        filters=F11,
        kernel_size=[1, 1],
        kernel_initializer=init,
        padding='same',
    )

    conv2d = conv2d_(A_prev)

    # batch_normalization
    batch_normalization_ = K.layers.BatchNormalization(
        axis=-1,
    )

    batch_normalization = batch_normalization_(conv2d)

    # activation layer using relu
    activation_ = K.layers.Activation(
        activation='relu'
    )

    activation = activation_(batch_normalization)

    # conv 3x3 + 1(S)
    conv2d_1_ = K.layers.Conv2D(
        filters=F3,
        kernel_size=[3, 3],
        kernel_initializer=init,
        padding='same',
    )

    conv2d_1 = conv2d_1_(activation)

    # batch_normalization
    batch_normalization_1_ = K.layers.BatchNormalization(
        axis=-1,
    )

    batch_normalization_1 = batch_normalization_1_(conv2d_1)

    # activation layer using relu
    activation_1_ = K.layers.Activation(
        activation='relu'
    )

    activation_1 = activation_1_(batch_normalization_1)

    # conv 1x1 + 1(S)
    conv2d_2_ = K.layers.Conv2D(
        filters=F12,
        kernel_size=[1, 1],
        kernel_initializer=init,
        padding='same',
    )

    conv2d_2 = conv2d_2_(activation_1)

    # batch_normalization
    batch_normalization_2_ = K.layers.BatchNormalization(
        axis=-1,
    )

    batch_normalization_2 = batch_normalization_2_(conv2d_2)

    # Add layer.
    # https://keras.io/api/layers/merging_layers/add/
    add_ = K.layers.Add()

    add = add_([batch_normalization_2, A_prev])

    # activation layer using relu
    activation_2_ = K.layers.Activation(
        activation='relu'
    )

    activation_2 = activation_2_(add)

    return activation_2
