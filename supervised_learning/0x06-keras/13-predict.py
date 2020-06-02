#!/usr/bin/env python3
"""
makes a prediction using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network
    :param network: the network model to make the prediction with
    :param data: the input data to make the prediction with
    :param verbose: is a boolean that determines if output should be printed
    during the prediction process
    :return: the prediction for the data
    """
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/
    # tf/keras/models/Sequential.md#predict
    return network.predict(x=data, verbose=verbose)
