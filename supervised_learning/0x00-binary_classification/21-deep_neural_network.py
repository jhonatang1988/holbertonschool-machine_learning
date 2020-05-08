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

    @staticmethod
    def _sigmoid_derivative(s):
        return s * (1 - s)

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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        one pass of the gradient descent function
        :param Y: (1, m) human outputs
        :param cache: it has the activations (A's)
        :param alpha: learning rate
        :return: nothing but updates __weights
        """
        weights_delta = {}

        # dg is the derivative of the activation function that in this case
        # is _sigmoid i put dg because g is the convention for the activation
        # function
        for i in range(self.__L, 0, -1):
            W = 'W' + str(i)
            b = 'b' + str(i)
            A = cache['A' + str(i)]
            A_left = cache['A' + str(i - 1)]
            if i == self.__L:
                dZ = A - Y
                dW = 1 / len(Y[0]) * (np.matmul(A_left, dZ.transpose()))
                db = 1 / len(Y[0]) * np.sum(dZ, axis=1, keepdims=True)

            else:
                dZ_right = dZ
                W_right = self.weights['W' + str(i + 1)]
                dg = self._sigmoid_derivative(A)
                dZ = (np.matmul(W_right.T, dZ_right)) * dg
                dW = 1 / len(Y[0]) * (np.matmul(dZ, A_left.transpose()))
                db = 1 / len(Y[0]) * (np.sum(dZ, axis=1, keepdims=True))

            weights_delta.update({self.__weights[W] - (alpha * dW)})
            weights_delta.update({self.__weights[b] - (alpha * db)})
            # self.weights[W] = self.weights[W] - alpha * dW
            # self.weights[b] = self.weights[b] - alpha * db

        self.__weights = weights_delta

        # el output siempre es distinto a las demas
        # A3 = cache['A3']
        # dZ3 = A3 - Y
        # dW3 = 1 / len(Y[0]) * (np.matmul(A2, dZ3.transpose()))
        # db3 = 1 / len(Y[0]) * np.sum(dZ3, axis=1, keepdims=True)

        # de aqui en adelante es igual, entonces se puede meter en un loop
        # dg2 = self._sigmoid_derivative(A2)
        # dZ2 = (np.matmul(self.weights['W3'].T, dZ3)) * dg2
        # dW2 = 1 / len(Y[0]) * (np.matmul(dZ2, A1.transpose()))
        # db2 = 1 / len(Y[0]) * (np.sum(dZ2, axis=1, keepdims=True))
        #
        # dg1 = self._sigmoid_derivative(A1)
        # dZ1 = (np.matmul(self.weights['W2'].T, dZ2)) * dg1
        # dW1 = 1 / len(Y[0]) * (np.matmul(dZ1, A0.transpose()))
        # db1 = 1 / len(Y[0]) * (np.sum(dZ1, axis=1, keepdims=True))

        # los pesos no se pueden actualizar en el loop porque por las
        # derivadas parciales se necesita que los pesos sigan igual. Se tiene
        # que crear una diccionario distinto para guardar el resultado
        # mientras el loop llega al final.

        # self.weights['W3'] = self.weights['W3'] - alpha * dW3.T
        # self.weights['b3'] = self.weights['b3'] - alpha * db3
        # self.weights['W2'] = self.weights['W2'] - alpha * dW2
        # self.weights['b2'] = self.weights['b2'] - alpha * db2
        # self.weights['W1'] = self.weights['W1'] - alpha * dW1
        # self.weights['b1'] = self.weights['b1'] - alpha * db1
