#!/usr/bin/env python3
"""
Neural network for binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


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

    @staticmethod
    def _sigmoid_derivative(s):
        return s * (1 - s)

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
        cost function. Average cost taking in account actions in last node __A2
        :param Y: shape (1, m) all human outputs
        :param A: shape (1, m)
        :return:
        """
        return -1 / len(Y[0]) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """
        as action function outputs between 0 and 1, like 0.323 it must be
        approximated to 1 or 0.
        :param X: input data
        :param Y: human outputs
        :return: action function outputs normalized to 1 or 0 and J (cost)
        """
        _, self.__A2 = self.forward_prop(X)
        A2_binary = np.where(self.__A2 >= 0.5, 1, 0)
        J = self.cost(Y, self.__A2)
        return A2_binary, J

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        the gradient descent function to low down weights and biases to their
        local minimum values
        https://www.youtube.com/watch?v=7bLEWDZng_M
        :param X: input data
        :param Y: human outputs
        :param A1: outputs of hidden layer
        :param A2: outputs of last layer (predictions)
        :param alpha: learning rate
        :return: nothing but updates __W1, __b1, __W2, and __b2
        """
        dZ2 = A2 - Y
        dW2 = 1 / len(Y[0]) * (np.matmul(A1, dZ2.transpose()))
        db2 = 1 / len(Y[0]) * np.sum(dZ2, axis=1, keepdims=True)

        # dg is the derivative of the activation function that in this case
        # is _sigmoid i put dg because g is the convention for the activation
        # function
        dg = self._sigmoid_derivative(A1)
        dZ1 = (np.matmul(self.__W2.transpose(), dZ2)) * dg
        dW1 = 1 / len(Y[0]) * (np.matmul(dZ1, X.transpose()))
        db1 = 1 / len(Y[0]) * (np.sum(dZ1, axis=1, keepdims=True))

        self.__W2 = self.__W2 - alpha * dW2.transpose()
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True,
              step=100):
        """
        function to loop the training set
        :param step: steps
        :param graph: to show the graph
        :param verbose: to show info every steps
        :param X: input data
        :param Y: human outputs
        :param iterations: number of iterations
        :param alpha: learning rate
        :return: the method self.evaluate after all iterations
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        elif iterations < 1:
            raise ValueError('iterations must be a positive integer')
        elif type(alpha) is not float:
            raise TypeError('alpha must be a float')
        elif alpha < 0:
            raise ValueError('alpha must be positive')
        elif verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not 1 < int(step) < int(iterations):
                raise ValueError('step must be positive and <= iterations')
            stepTemp = step
            cost_each_iteration = {}

        for iteration in range(iterations):
            A1, A2 = self.forward_prop(X)
            if iteration == 0 and verbose:
                J = self.cost(Y, self.__A2)
                if graph:
                    cost_each_iteration[0] = J
                print('Cost after {} iterations: {}'.format(
                    iteration, J))
            self.gradient_descent(X, Y, A1, A2, alpha)
            if iteration == step and verbose:
                J = self.cost(Y, self.__A2)
                if graph:
                    cost_each_iteration[step] = J
                print('Cost after {} iterations: {}'.format(
                    iteration, J))
                step += stepTemp

        if verbose:
            J = self.cost(Y, self.__A2)
            if graph:
                cost_each_iteration[iterations] = J
            print('Cost after {} iterations: {}'.format(
                iterations, J))

        if graph:
            lists_for_plot = sorted(cost_each_iteration.items())
            x, y = zip(*lists_for_plot)
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        _, self.__A2 = self.forward_prop(X)
        A2_binary = np.where(self.__A2 >= 0.5, 1, 0)
        J = self.cost(Y, self.__A2)

        return A2_binary, J
