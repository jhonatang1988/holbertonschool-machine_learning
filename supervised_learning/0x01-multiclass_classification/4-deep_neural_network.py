#!/usr/bin/env python3
"""
Deep Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Deep Neural Network class
    """

    def __init__(self, nx, layers, activation='sig'):
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
        elif activation != 'sigmoid' or activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
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
    def activation(self):
        return self.__activation

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
    def _softMax(Z):
        exps = np.exp(Z - np.max(Z))
        return exps / np.sum(exps, axis=0, keepdims=True)

    @staticmethod
    def _sigmoid(Z):
        """
        sigmoid function
        :param Z: loss function
        :return: sigmoid function of loss function
        """
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _sigmoid_derivative(A):
        """
        _sigmoid_derivative for the backpropagation
        :param s: sigmoid (A)
        :return: derivative of sigmoid
        """
        return A * (1 - A)

    @staticmethod
    def _tanh(Z):
        """
        tanh function
        :param z: loss function
        :return: tanh function of loss function
        """
        return np.tanh(Z)

    @staticmethod
    def _tanh_derivative(A):
        """
        derivative of tanh function
        :param A: tanh (A)
        :return: derivative of tanh
        """
        return 1 - A ** 2

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
            Z = np.matmul(self.__weights[key_search_weights],
                          self.__cache[key_earlier_activations]) + \
                self.__weights[key_search_biases]
            if i == self.__L:
                value_for_activations = self._softMax(Z)
            else:
                if self.__activation == 'sig':
                    value_for_activations = self._sigmoid(Z)
                elif self.__activation == 'tanh':
                    value_for_activations = self._tanh(Z)

            self.__cache.update({key_for_activations: value_for_activations})

        return value_for_activations, self.__cache

    @staticmethod
    def cost(Y, A):
        """
        cost function for the network
        :param Y: shape (1, m) with the input data
        :param A: shape (1, m)  with activations outputs
        :return: cost (J)
        """
        m = Y.shape[1]
        log_likelihood = -(Y * np.log(A))
        return np.sum(log_likelihood) / m

    def evaluate(self, X, Y):
        """
        evaluates network for one pass
        :param X: shape (nx, m) with input data
        :param Y: shape (1, m) with human outputs
        :return: activations for last layer (predictions) and cost (J)
        """
        A, _ = self.forward_prop(X)
        A_multi_class = np.where(np.amax(A, axis=0), 1, 0)
        J = self.cost(Y, A)
        return A_multi_class, J

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
                weights_delta.update({W: self.weights[W] - (
                        alpha * dW.T)})
                weights_delta.update({b: self.weights[b] - (
                        alpha * db)})

            else:
                dZ_right = dZ
                W_right = self.weights['W' + str(i + 1)]
                if self.__activation == 'sig':
                    dg = self._sigmoid_derivative(A)
                elif self.__activation == 'tanh':
                    dg = self._tanh_derivative(A)
                dZ = (np.matmul(W_right.T, dZ_right)) * dg
                dW = 1 / len(Y[0]) * (np.matmul(dZ, A_left.transpose()))
                db = 1 / len(Y[0]) * (np.sum(dZ, axis=1, keepdims=True))

                weights_delta.update({W: self.weights[W] - (
                        alpha * dW)})
                weights_delta.update({b: self.weights[b] - (
                        alpha * db)})

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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        the loops over the input data
        :param verbose: to show evaluation info on each step
        :param step: steps or minibatches
        :param graph: to show a graph of the results
        :param X: (nx, m) with input data
        :param Y: (1, m) with human outputs
        :param iterations: loops
        :param alpha: learning rate
        :return: evaluate network after iterations and updates __weights and
        __cache
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        elif iterations < 1:
            raise ValueError('iterations must be a positive integer')
        elif type(alpha) is not float:
            raise TypeError('alpha must be a float')
        elif alpha <= 0:
            raise ValueError('alpha must be positive')
        elif verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not 1 < int(step) < int(iterations):
                raise ValueError('step must be positive and <= iterations')
            stepTemp = step
            cost_each_iteration = {}

        for iteration in range(iterations):
            A, cache = self.forward_prop(X)
            if iteration == 0 and verbose:
                J = self.cost(Y, A)
                if graph:
                    cost_each_iteration[0] = J
                print('Cost after {} iterations: {}'.format(
                    iteration, J))
            self.gradient_descent(Y, cache, alpha)
            if iteration == step and verbose:
                J = self.cost(Y, A)
                if graph:
                    cost_each_iteration[step] = J
                print('Cost after {} iterations: {}'.format(
                    iteration, J))
                step += stepTemp

        if verbose:
            J = self.cost(Y, A)
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

        A, _ = self.forward_prop(X)
        A_binary = np.where(A >= 0.5, 1, 0)
        J = self.cost(Y, A)
        return A_binary, J

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        :param filename: is the file to which the object should be saved
        :return: nothing
        """
        if not filename:
            return None
        extension = filename[-4:]
        if extension != '.pkl':
            filename += '.pkl'
        fileObj = open(filename, 'wb')
        pickle.dump(self, fileObj)
        fileObj.close()

    @staticmethod
    def load(filename):
        """
        loads a DeepNeuralNetwork object
        :param filename: fileObj
        :return: DeepNeuralNetwork object
        """
        if not filename:
            return None
        try:
            fileObj = open(filename, 'rb')
        except OSError:
            return None
        return pickle.load(fileObj)
