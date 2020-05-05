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

    def cost(self, Y, A):
        """
        function cost J is the convention
        :param Y: real outputs
        :param A: guess of the output, known as A[2] in NN
        :return: J or cost function
        """
        return -1 / len(Y[0]) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 -
                                              A)))

    def evaluate(self, X, Y):
        """
        direct method to get A and J directly
        :param X: input data5
        :param Y: human outputs
        :return: A and J
        """
        A = self.forward_prop(X)
        # A_binary is the activation function convert to 0 or 1 because there
        # are only two possible outputs
        A_binary = np.where(A >= 0.5, 1, 0)
        J = self.cost(Y, A)
        return A_binary, J

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        one pass of the gradient descent function to the NN
        :param X: input data (nx, m)
        :param Y: human outputs (1, m)
        :param A: predictions 1 or 0 in shape (1, m)
        :param alpha: learning rate
        :return: nothing but updates __W and __b just one time
        """
        dZ = A - Y
        dW = 1 / len(Y[0]) * (np.matmul(X, dZ.transpose()))
        self.__W = self.__W - alpha * dW.transpose()
        dB = 1 / len(Y[0]) * np.sum(dZ)
        self.__b = self.__b - alpha * dB

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        trains the neurons
        :param X: input data
        :param Y: human outputs
        :param iterations: number of iterations
        :param alpha: learning rate
        :return: self.evaluate function
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        elif iterations < 1:
            raise ValueError('iterations must be a positive integer')
        elif type(alpha) is not float:
            raise TypeError('alpha must be a float')
        elif alpha < 0:
            raise ValueError('alpha must be positive')
        else:
            for i in range(iterations):
                self.__A = self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
            return self.evaluate(X, Y)
