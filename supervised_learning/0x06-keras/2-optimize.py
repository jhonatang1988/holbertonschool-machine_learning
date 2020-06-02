#!/usr/bin/env python3
"""
sets up Adam optimization for a keras model with categorical crossentropy
loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metrics
    :param network: the model to optimize
    :param alpha: the learning rate
    :param beta1: the first Adam optimization parameter
    :param beta2: the second Adam optimization parameter
    :return: None
    """
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
