#!/usr/bin/env python3
"""
 builds the inception network as described in Going Deeper with Convolutions
 https://arxiv.org/pdf/1409.4842.pdf
"""
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in Going Deeper with Convolutions
    :return: the keras model
    """
    init = K.initializers.he_normal()
    input_1 = K.Input(shape=(224, 224, 3))

    # layer 1, in the design would be: Conv 7x7 + 2(S)
    conv2d_ = K.layers.Conv2D(
        filters=64,
        kernel_initializer=init,
        activation='relu',
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same'
    )
    conv2d = conv2d_(input_1)

    # layer 1, MaxPool 3x3 + 2(S)
    max_pooling2d_ = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )

    max_pooling2d = max_pooling2d_(conv2d)

    # layer 3, Conv 1x1 + 1(V)
    conv2d_1_ = K.layers.Conv2D(
        filters=64,
        kernel_initializer=init,
        activation='relu',
        kernel_size=[1, 1],
        strides=(1, 1),
        padding='valid'
    )

    conv2d_1 = conv2d_1_(max_pooling2d)

    # layer 4, Conv 3x3+1(S)
    conv2d_2_ = K.layers.Conv2D(
        filters=192,
        kernel_initializer=init,
        activation='relu',
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same'
    )

    conv2d_2 = conv2d_2_(conv2d_1)

    # layer 5, MaxPool 3x3+2(S)
    max_pooling2d_1_ = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )

    max_pooling2d_1 = max_pooling2d_1_(conv2d_2)

    # inception_block 1 [64, 96, 128, 16, 32, 32]
    concatenate = inception_block(max_pooling2d_1,
                                  [64, 96, 128, 16, 32, 32])

    # inception_block 1 [128, 128, 192, 32, 32, 64]
    concatenate_1 = inception_block(concatenate,
                                    [128, 128, 192, 32, 32, 64])

    # MaxPool 3x3+2(S)
    max_pooling2d_4_ = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )

    max_pooling2d_4 = max_pooling2d_4_(concatenate_1)

    # inception_block [192, 96, 208, 16, 48, 64]
    concatenate_2 = inception_block(max_pooling2d_4,
                                    [192, 96, 208, 16, 48, 64])

    # inception_block [160, 112, 224, 24, 64, 64]
    concatenate_3 = inception_block(concatenate_2,
                                    [160, 112, 224, 24, 64, 64])

    # inception_block [128, 128, 256, 24, 64, 64]
    concatenate_4 = inception_block(concatenate_3,
                                    [128, 128, 256, 24, 64, 64])

    # inception_block [112, 144, 288, 32, 64, 64]
    concatenate_5 = inception_block(concatenate_4,
                                    [112, 144, 288, 32, 64, 64])

    # inception_block [256, 160, 320, 32, 128, 128]
    concatenate_6 = inception_block(concatenate_5,
                                    [256, 160, 320, 32, 128, 128])

    # MaxPool 3x3+2(S)
    max_pooling2d_10_ = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )

    max_pooling2d_10 = max_pooling2d_10_(concatenate_6)

    # inception_block [256, 160, 320, 32, 128, 128]
    concatenate_7 = inception_block(max_pooling2d_10,
                                    [256, 160, 320, 32, 128, 128])

    # inception_block [384, 192, 384, 48, 128, 128]
    concatenate_8 = inception_block(concatenate_7,
                                    [384, 192, 384, 48, 128, 128])

    # AveragePool 7x7+1(V)
    # https://keras.io/api/layers/pooling_layers/average_pooling2d/
    average_pooling2d_ = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(2, 2),
        padding='valid'
    )

    average_pooling2d = average_pooling2d_(concatenate_8)

    # dropout (40%)
    dropout_ = K.layers.Dropout(rate=.4)

    dropout = dropout_(average_pooling2d)

    # non linear Fully connected
    dense_ = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )
    dense = dense_(dropout)

    # with softmax (vector of categorical probabilities)
    # softmax = K.activations.softmax(dense)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
