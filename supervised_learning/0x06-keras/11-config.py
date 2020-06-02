#!/usr/bin/env python3
"""
saves and loads a model’s configuration in JSON format
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    :param network: the model whose configuration should be saved
    :param filename: the path of the file that the configuration should be
    saved to
    :return: None
    """
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_as_json = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(model_as_json)

    return None


def load_config(filename):
    """
    loads a model with a specific configuration
    :param filename: the path of the file containing the model’s configuration
    in JSON format
    :return: the loaded model
    """
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    return K.models.model_from_json(loaded_model_json)
