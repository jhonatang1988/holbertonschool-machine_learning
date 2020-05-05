#!/usr/bin/env python3
"""
Neuron Class
"""
import numpy as np


class Neuron:
    """
    Neuron for binary classification
    """

    def __init__(self, nx):
        """
        __init__
        :param nx: number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.randn(1, nx)
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """
        W for Weights
        :return: private Weights
        """
        return self.__W

    @property
    def b(self):
        """
        biases getter
        :return: private biases
        """
        return self.__b

    @property
    def A(self):
        """
        A for Activation getter
        :return: private activation
        """
        return self.__A

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
        function to apply to input features
        :param X: input data
        :return: activation function stored in _A
        """
        self.__A = self._sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A
