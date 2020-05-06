#!/usr/bin/env python3
"""
Neural network for binary classification
"""


class NeuralNetwork:
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        elif nodes < 1:
            raise ValueError('nodes must be a positive integer')
