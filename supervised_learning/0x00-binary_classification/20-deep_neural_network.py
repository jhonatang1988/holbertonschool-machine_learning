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

        self.__L = len(layers)
        if self.__L == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__cache = {}
        # initialized using He et al. w=np.random.randn(layer_size[l],
        # layer_size[l-1])*np.sqrt(2/layer_size[l-1])
        self.__weights = {}
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
            self.__weights[key_for_weights] = value_for_weights

            key_for_biases = 'b' + str(i)
            value_for_biases = np.zeros((layers[i], 1))
            self.__weights[key_for_biases] = value_for_biases
        del layers_with_inputs[0]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def _sigmoid(z):
        """
        sigmoid function
        :param z: loss function
        :return: sigmoid function of loss function
        """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """
        forward propagation of the network - just one pass
        :param X: input data (nx, m) for this exercise (784, 46...something)
        :return: the output of the network (A3 for this exercise) and __cache
        """
        # self.__A1 = self._sigmoid(np.matmul(self.__W1, X) + self.__b1)
        # we add the inputs to the cache as inputs could be considered as
        # activations for the earlier layer
        self.__cache.update({'A0': X})
        for i in range(1, self.__L + 1):
            key_for_activations = 'A' + str(i)
            key_earlier_activations = 'A' + str(i - 1)
            key_search_weights = 'W' + str(i)
            key_search_biases = 'b' + str(i)
            value_for_activations = self._sigmoid(
                np.matmul(self.__weights[key_search_weights],
                          self.__cache[key_earlier_activations]) +
                self.__weights[key_search_biases])
            self.__cache.update({key_for_activations: value_for_activations})

        return value_for_activations, self.__cache

    def cost(self, Y, A):
        """
        cost function for the network
        :param Y: shape (1, m) with the input data
        :param A: shape (1, m)  with activations outputs
        :return: cost (J)
        """
        return -1 / len(Y[0]) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """
        evaluates network for one pass
        :param X: shape (nx, m) with input data
        :param Y: shape (1, m) with human outputs
        :return: activations for last layer (predictions) and cost (J)
        """
        A, _ = self.forward_prop(X)
        A_binary = np.where(A >= 0.5, 1, 0)
        J = self.cost(Y, A)
        return A_binary, J
