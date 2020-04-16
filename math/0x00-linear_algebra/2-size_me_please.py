#!/usr/bin/env python3
"""
wrapper to keep state of shape
"""


def _matrix_shape(matrix):

    global shape
    for i in matrix:
        if type(i) is list:
            shape.append(len(i))
            _matrix_shape(i)
        return shape


"""
the main function
"""


def matrix_shape(matrix):
    global shape
    shape = [len(matrix)]
    return _matrix_shape(matrix)
