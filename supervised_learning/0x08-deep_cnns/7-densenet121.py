#!/usr/bin/env python3
"""
builds the DenseNet-121 architecture as described in Densely Connected
Convolutional Networks
https://arxiv.org/pdf/1608.06993.pdf
"""
import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks
    :param growth_rate: is the growth rate
    :param compression: is the compression factor
    :return: the keras model
    """
    init = K.initializers.he_normal()
    input_1 = K.Input(shape=(224, 224, 3))

    batch_normalization_ = K.layers.BatchNormalization(
        axis=-1
    )

    batch_normalization = batch_normalization_(input_1)

    activation_ = K.layers.Activation(
        activation='relu'
    )

    activation = activation_(batch_normalization)

    conv2d_ = K.layers.Conv2D(
        filters=64,
        kernel_size=[7, 7],
        kernel_initializer=init,
        padding='same',
        strides=(2, 2)
    )

    conv2d = conv2d_(activation)

    # maxpoling
    max_pooling2d_ = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )

    max_pooling2d = max_pooling2d_(conv2d)

    concatenate_5, nb_filters = dense_block(max_pooling2d, 64, growth_rate, 6)
    average_pooling2d, nb_filters = transition_layer(concatenate_5,
                                                     nb_filters, compression)
    concatenate_17, nb_filters = dense_block(average_pooling2d, nb_filters,
                                             growth_rate, 12)

    average_pooling2d_1, nb_filters = transition_layer(concatenate_17,
                                                       nb_filters, compression)

    concatenate_41, nb_filters = dense_block(average_pooling2d_1, nb_filters,
                                             growth_rate, 24)

    average_pooling2d_2, nb_filters = transition_layer(concatenate_41,
                                                       nb_filters,
                                                       compression)

    concatenate_57, _ = dense_block(average_pooling2d_2, nb_filters,
                                    growth_rate, 16)

    average_pooling2d_3_ = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )

    average_pooling2d_3 = average_pooling2d_3_(concatenate_57)

    # non linear Fully connected
    dense_ = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )
    dense = dense_(average_pooling2d_3)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
