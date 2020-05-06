#!/usr/bin/env python3
"""
Neural network for binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class
    """

    def __init__(self, nx, nodes):
        """
        init NeuralNetwork instance
        :param nx: number of inputs
        :param nodes: number of layers
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        elif nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

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
        the activation function
        :param X: input data
        :return: updates __A1 and __A2 and returns them
        """
        self.__A1 = self._sigmoid(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self._sigmoid(np.matmul(self.__W2, self.__A1) + self.__b2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        cost funcion. Average cost taking in account actions in last node __A2
        :param Y: shape (1, m) all human outputs
        :param A: shape (1, m)
        :return:
        """
        return -1 / len(Y[0]) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
