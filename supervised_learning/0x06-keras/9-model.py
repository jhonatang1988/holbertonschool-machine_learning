#!/usr/bin/env python3
"""
saves and load an entire model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    :param network: the model to save
    :param filename: the path of the file that the model should be saved to
    :return: None
    """
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python
    # /tf/keras/models/save_model.md
    K.models.save_model(
        model=network,
        filepath=filename
    )

    return None


def load_model(filename):
    """
    loads an entire model
    :param filename: the path of the file that the model should be loaded from
    :return: the loaded model
    """
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python
    # /tf/keras/models/load_model.md
    return K.models.load_model(
        filepath=filename
    )
