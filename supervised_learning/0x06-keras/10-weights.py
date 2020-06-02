#!/usr/bin/env python3
"""
saves and loads a model’s weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model’s weights
    :param network: the model whose weights should be saved
    :param filename: the path of the file that the weights should be saved to
    :param save_format: the format in which the weights should be saved
    :return: None
    """
    # https://keras.io/getting_started/faq/
    if filename[-3:] != '.h5':
        filename += ('.' + save_format)

    network.save_weights(filename)

    return None


def load_weights(network, filename):
    """
    loads a model’s weights
    :param network: the model to which the weights should be loaded
    :param filename: the path of the file that the weights should be loaded
    from
    :return: None
    """
    # https://keras.io/getting_started/faq/
    network.load_weights(filename)

    return None
