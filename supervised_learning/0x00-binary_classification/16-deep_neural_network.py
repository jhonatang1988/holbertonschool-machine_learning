#!/usr/bin/env python3
"""
Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network class
    """

    def __init__(self, nx, layers):
        """
        init method
        :param nx: number of inputs
        :param layers: list [] of layers.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        if self.L == 0:
            raise TypeError('layers must be a list of positive integers')
        self.cache = {}
        # initialized using He et al. w=np.random.randn(layer_size[l],
        # layer_size[l-1])*np.sqrt(2/layer_size[l-1])
        self.weights = {}
        # we add the inputs at the start of the layer list to be able to
        # create the He et al. method
        layers_with_inputs = layers
        layers_with_inputs.insert(0, nx)
        for i in range(1, len(layers_with_inputs)):
            if type(layers_with_inputs[i]) is not int \
                    or layers_with_inputs[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            key_for_weights = 'W' + str(i)
            value_for_weights = np.random.randn(layers_with_inputs[i],
                                                layers_with_inputs[
                                                    i - 1]) * np.sqrt(
                2 / layers_with_inputs[i - 1])
            self.weights[key_for_weights] = value_for_weights

            key_for_biases = 'b' + str(i)
            value_for_biases = np.zeros((layers_with_inputs[i], 1))
            self.weights[key_for_biases] = value_for_biases
