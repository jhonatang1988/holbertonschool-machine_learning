#!/usr/bin/env python3
"""
builds the ResNet-50 architecture as described in Deep Residual Learning
for Image Recognition (2015)
https://arxiv.org/pdf/1512.03385.pdf
input_1
conv2d
batch_normalization
activation
max_pooling2d

projection
identity
identity

projection
identity
identity
identity

projection
identity
identity
identity
identity
identity

projection
identity
identity

average_pooling2d
dense
"""
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015)
    :return: the keras model
    """
    init = K.initializers.he_normal()
    input_1 = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same', strides=(2, 2),
                            kernel_initializer=init)(input_1)

    # batch_normalization
    batch_normalization_ = K.layers.BatchNormalization(
        axis=-1,
    )

    batch_normalization = batch_normalization_(conv1)

    # activation layer using relu
    activation_ = K.layers.Activation(
        activation='relu'
    )

    activation = activation_(batch_normalization)

    # maxpoling
    max_pooling2d_ = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )

    max_pooling2d = max_pooling2d_(activation)

    # primer bloque: cada bloque es un projection al principio y un numero
    # de identityBlock determinado. (1 projection_block + 2 identity_block)
    activation_3 = projection_block(max_pooling2d, [64, 64, 256], 1)
    activation_6 = identity_block(activation_3, [64, 64, 256])
    activation_9 = identity_block(activation_6, [64, 64, 256])

    # segundo bloque: (1 projection_block + 3 identity_block)
    activation_12 = projection_block(activation_9, [128, 128, 512])
    activation_15 = identity_block(activation_12, [128, 128, 512])
    activation_18 = identity_block(activation_15, [128, 128, 512])
    activation_21 = identity_block(activation_18, [128, 128, 512])

    # tercer bloque: (1 projection_block + 5 identity_block)
    activation_24 = projection_block(activation_21, [256, 256, 1024])
    activation_27 = identity_block(activation_24, [256, 256, 1024])
    activation_30 = identity_block(activation_27, [256, 256, 1024])
    activation_33 = identity_block(activation_30, [256, 256, 1024])
    activation_36 = identity_block(activation_33, [256, 256, 1024])
    activation_39 = identity_block(activation_36, [256, 256, 1024])

    # cuarto bloque: (1 projection_block + 2 identity_block)
    activation_42 = projection_block(activation_39, [512, 512, 2048])
    activation_45 = identity_block(activation_42, [512, 512, 2048])
    activation_48 = identity_block(activation_45, [512, 512, 2048])

    # AveragePool 7x7+1(V)
    # https://keras.io/api/layers/pooling_layers/average_pooling2d/
    average_pooling2d_ = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )

    average_pooling2d = average_pooling2d_(activation_48)

    # non linear Fully connected
    dense_ = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )
    dense = dense_(average_pooling2d)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
