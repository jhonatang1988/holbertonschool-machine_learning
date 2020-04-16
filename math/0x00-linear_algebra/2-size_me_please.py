#!/usr/bin/env python3
def _matrix_shape(matrix):
    global shape
    for i in matrix:
        if type(i) is list:
            shape.append(len(i))
            _matrix_shape(i)
        return shape


def matrix_shape(matrix):
    global shape
    shape = [len(matrix)]
    return _matrix_shape(matrix)
