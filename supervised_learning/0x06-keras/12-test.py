#!/usr/bin/env python3
"""
that tests a neural network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    that tests a neural network
    :param network: the network model to test
    :param data: the input data to test the model with
    :param labels: the correct one-hot labels of data
    :param verbose: a boolean that determines if output should be printed
    during the testing process
    :return: the loss and accuracy of the model with the testing data,
    respectively
    """
    score = network.evaluate(data, labels, verbose=verbose)

    return score
